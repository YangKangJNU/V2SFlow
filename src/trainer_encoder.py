import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils import default, exists
from adam_atan2_pytorch.adopt import Adopt
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.nn.functional as F

class LRS3Dataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.filelist, self.f_names = self.get_filelist(self.root, self.mode)
        self.hubert_unit_dict = self.build_hubert_unit_dict()
        self.pitch_unit_dict = self.build_pitch_unit_dict()

    def build_hubert_unit_dict(self):
        base_fname_batch, quantized_units_batch = [], []
        with open(f"{self.root}/hubert_units") as f:
            for line in f:
                base_fname, quantized_units_str = line.rstrip().split("|")
                quantized_units = [int(q) for q in quantized_units_str.split(" ")]
                base_fname_batch.append(base_fname)
                quantized_units_batch.append(quantized_units)
        self.unit_dict = dict(zip(base_fname_batch,quantized_units_batch))
    def build_pitch_unit_dict(self):
        base_fname_batch, quantized_units_batch = [], []
        with open(f"{self.root}/pitch_units") as f:
            for line in f:
                base_fname, quantized_units_str = line.rstrip().split("|")
                quantized_units = [int(q) for q in quantized_units_str.split(" ")]
                base_fname_batch.append(base_fname)
                quantized_units_batch.append(quantized_units)
        self.unit_dict = dict(zip(base_fname_batch,quantized_units_batch))

    def get_filelist(self, root, mode):
        file_list, paths = [], []
        with open(f"{root}/433h_data/train.tsv", "r") as f:
            train_data = f.readlines()[1:]
        for i in range(len(train_data)):
            file = train_data[i].split('\t')[0]
            file_list.append(file)
            paths.append(f"{root}/video/{file}")
        return paths, file_list

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        file_path = self.filelist[index]
        f_name = self.f_names[index]
        feature_path = file_path.replace('/video/', '/feature/') + '.npy'
        feat = torch.tensor(np.load(feature_path)).squeeze(0)
        spk_emb_path = file_path.replace('/video/', '/spk_emb/') + '.pt'
        spk_emb = torch.load(spk_emb_path).squeeze(0)
        hubert_unit = torch.tensor(self.hubert_unit_dict[f_name]) + 1
        pitch_unit = torch.tensor(self.pitch_unit_dict[f_name]) + 1

        diff = feat.size(0) - len(hubert_unit)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=hubert_unit.dtype)
            hubert_unit = torch.cat((hubert_unit, padding))
        elif diff > 0:
            hubert_unit = hubert_unit[:-diff]

        diff = feat.size(0) - len(pitch_unit)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=pitch_unit.dtype)
            pitch_unit = torch.cat((pitch_unit, padding))
        elif diff > 0:
            pitch_unit = pitch_unit[:-diff] 

        return feat, spk_emb, hubert_unit, pitch_unit, f_name

def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(0) for x in batch]),
        dim=0, descending=True)
    
    max_feat_len = max([x[0].size(0) for x in batch])
    max_hubert_len = max([x[2].size(0) for x in batch])
    max_pitch_len = max([x[3].size(0) for x in batch])

    feat_lengths = torch.IntTensor(len(batch))
    hubert_lengths = torch.IntTensor(len(batch))
    pitch_lengths = torch.IntTensor(len(batch))

    feat_padded = torch.zeros(len(batch), max_feat_len, 1024, dtype=torch.float32)
    hubert_padded = torch.zeros(len(batch), max_hubert_len, dtype=torch.int64)
    pitch_padded = torch.zeros(len(batch), max_pitch_len, dtype=torch.int64)
    
    feat_padded.zero_()
    hubert_padded.zero_()
    pitch_padded.zero_()

    spk_emb = []
    file_name = []

    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        feat = row[0]
        feat_padded[i, :row[0].size(0), :] = feat
        feat_lengths[i] = row[0].size(0)

        spk_emb.append(row[1])

        hubert_unit = row[2]
        hubert_padded[i, :row[2].size(0)] = hubert_unit
        hubert_lengths[i] = row[2].size(0)

        pitch = row[3]
        pitch_padded[i, :row[3].size(0)] = pitch
        pitch_lengths[i] = row[3].size(0)

        file_name.append(row[4])
    
    spk_emb = torch.stack(spk_emb)

    return dict(
        avhubert_feat = feat_padded,
        avhubert_feat_len = feat_lengths,
        hubert_unit = hubert_padded,
        hubert_unit_len = hubert_lengths,
        pitch_unit = pitch_padded,
        pitch_unit_len = pitch_lengths,
        spk_emb = spk_emb,
        file_name = file_name
    )   

class Trainer:
    def __init__(
            self,
            content_encoder = None,
            pitch_encoder = None,
            spk_emb_encoder = None,
            lr = 1e-4,
            checkpoint_path = None,
            grad_accumulation_steps = 1,
            log_file = 'logs.txt',
            tensorboard_log_dir = 'logs/test1',
            accelerate_kwargs: dict = dict(),
            num_warmup_steps = None,
            sample_steps = 50,
            max_grad_norm = 1.0,
    ):
        logger.add(log_file)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            mixed_precision='fp16',
            **accelerate_kwargs
        )        
        self.content_encoder = content_encoder
        self.pitch_encoder = pitch_encoder
        self.spk_emb_encoder = spk_emb_encoder
        self.lr = lr
        self.ema_model = None
        self.optimieze = None
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = default(checkpoint_path, 'model.pth') 
        self.sample_steps = sample_steps
        self.tensorboard_log_dir = tensorboard_log_dir

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.scheduler = None

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def load_checkpoint(self, ckpt_path, model):
        if not exists(ckpt_path) or not os.path.exists(ckpt_path):
            return 0

        checkpoint = torch.load(ckpt_path)
        self.accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step'] 
    
    def save_checkpoint(self, step, ckpt_name, model, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict = self.scheduler.state_dict(),
                step = step
            )

            self.accelerator.save(checkpoint, ckpt_name)

    def train_content_encoder(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, save_step=2000, val_step=1000):
        self.optimizer = Adopt(self.content_encoder.parameters(), lr = self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        self.content_encoder, train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(self.content_encoder, train_dataloader, val_dataloder, self.scheduler)
        start_step = self.load_checkpoint(ckpt_path, self.content_encoder)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

        for epoch in range(start_epoch, epochs):
            self.content_encoder.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.content_encoder):            
                    avhubert_feat = batch['avhubert_feat']
                    avhubert_feat_len = batch['avhubert_feat_len']
                    hubert_unit = batch['hubert_unit']
                    logits = self.content_encoder(avhubert_feat, avhubert_feat_len)
                    ce_loss = ce_loss_fn(logits.permute(0, 2, 1), hubert_unit-1)
                    acc = ((logits.argmax(dim=-1) == hubert_unit-1).sum() / ((hubert_unit-1) != -1).sum()).item()
                    self.accelerator.backward(ce_loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.content_encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:        
                        logger.info(f"step {global_step+1}: ce_loss = {ce_loss.item():.4f}, acc = {np.mean(acc):.4f}")
                        self.writer.add_scalar('train_content/ce_loss', ce_loss.item(), global_step)
                        self.writer.add_scalar('train_content/acc', acc, global_step)

                if global_step % val_step == 0:
                    eval_acc = self.evaluate_content_encoder(val_dataloder, global_step)
                    logger.info(f"step {global_step+1}: eval_acc = {eval_acc:.4f}s")

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}.pt'.format(global_step, eval_acc)), self.content_encoder)

                global_step += 1
                epoch_loss += ce_loss.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()

    def evaluate_content_encoder(self, val_dataloader, global_step):
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        accs = []
        ce_losses = []
        self.content_encoder.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                avhubert_feat = batch['avhubert_feat']
                avhubert_feat_len = batch['avhubert_feat_len']
                hubert_unit = batch['hubert_unit']
                logits = self.content_encoder(avhubert_feat, avhubert_feat_len)
                ce_loss = ce_loss_fn(logits.permute(0, 2, 1), hubert_unit-1)
                acc = ((logits.argmax(dim=-1) == hubert_unit-1).sum() / ((hubert_unit-1) != -1).sum()).item()
                accs.append(acc)
                ce_losses.append(ce_loss.item())
        self.writer.add_scalar(f"eval_content/acc", np.mean(accs), global_step)
        self.writer.add_scalar(f"eval_content/ce_loss", np.mean(ce_losses), global_step)
        self.content_encoder.train()
        return np.mean(accs)       
                    
    def train_pitch_encoder(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, save_step=2000, val_step=1000):
        self.optimizer = Adopt(self.pitch_encoder.parameters(), lr = self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        self.pitch_encoder, train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(self.pitch_encoder, train_dataloader, val_dataloder, self.scheduler)
        start_step = self.load_checkpoint(ckpt_path, self.pitch_encoder)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

        for epoch in range(start_epoch, epochs):
            self.pitch_encoder.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.pitch_encoder):            
                    avhubert_feat = batch['avhubert_feat']
                    avhubert_feat_len = batch['avhubert_feat_len']
                    pitch_unit = batch['pitch_unit']
                    logits = self.pitch_encoder(avhubert_feat, avhubert_feat_len)
                    ce_loss = ce_loss_fn(logits.permute(0, 2, 1), pitch_unit-1)
                    acc = ((logits.argmax(dim=-1) == pitch_unit-1).sum() / ((pitch_unit-1) != -1).sum()).item()
                    self.accelerator.backward(ce_loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.pitch_encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:        
                        logger.info(f"step {global_step+1}: ce_loss = {ce_loss.item():.4f}, acc = {np.mean(acc):.4f}")
                        self.writer.add_scalar('train_pitch/ce_loss', ce_loss.item(), global_step)
                        self.writer.add_scalar('train_pitch/acc', acc, global_step)

                if global_step % val_step == 0:
                    eval_acc = self.evaluate_pitch_encoder(val_dataloder, global_step)
                    logger.info(f"step {global_step+1}: eval_acc = {eval_acc:.4f}s")

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}.pt'.format(global_step, eval_acc)), self.pitch_encoder)

                global_step += 1
                epoch_loss += ce_loss.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()

    def evaluate_pitch_encoder(self, val_dataloader, global_step):
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        accs = []
        ce_losses = []
        self.pitch_encoder.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                avhubert_feat = batch['avhubert_feat']
                avhubert_feat_len = batch['avhubert_feat_len']
                pitch_unit = batch['pitch_unit']
                logits = self.content_encoder(avhubert_feat, avhubert_feat_len)
                ce_loss = ce_loss_fn(logits.permute(0, 2, 1), pitch_unit-1)
                acc = ((logits.argmax(dim=-1) == pitch_unit-1).sum() / ((pitch_unit-1) != -1).sum()).item()
                accs.append(acc)
                ce_losses.append(ce_loss.item())
        self.writer.add_scalar(f"eval_pitch/acc", np.mean(accs), global_step)
        self.writer.add_scalar(f"eval_pitch/ce_loss", np.mean(ce_losses), global_step)
        self.pitch_encoder.train()
        return np.mean(accs)

    def train_spk_encoder(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, save_step=2000, val_step=1000):
        self.optimizer = Adopt(self.spk_emb_encoder.parameters(), lr = self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        self.spk_emb_encoder, train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(self.spk_emb_encoder, train_dataloader, val_dataloder, self.scheduler)
        start_step = self.load_checkpoint(ckpt_path, self.spk_emb_encoder)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1
        cs_loss_fn = torch.nn.CosineEmbeddingLoss()

        for epoch in range(start_epoch, epochs):
            self.spk_emb_encoder.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.spk_emb_encoder):            
                    avhubert_feat = batch['avhubert_feat']
                    avhubert_feat_len = batch['avhubert_feat_len']
                    spk_emb = batch['spk_emb']
                    spk_emb_hat = self.spk_emb_encoder(avhubert_feat, avhubert_feat_len)
                    cs_loss = cs_loss_fn(spk_emb_hat, spk_emb, torch.ones(spk_emb_hat.size(0)))
                    similarity = F.cosine_similarity(spk_emb_hat, spk_emb)
                    self.accelerator.backward(cs_loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.spk_emb_encoder.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:        
                        logger.info(f"step {global_step+1}: cs_loss = {cs_loss.item():.4f}, acc = {np.mean(similarity):.4f}")
                        self.writer.add_scalar('train_spk/cs_loss', cs_loss.item(), global_step)
                        self.writer.add_scalar('train_spk/similarity', similarity, global_step)

                if global_step % val_step == 0:
                    eval_similarity = self.evaluate_spk_encoder(val_dataloder, global_step)
                    logger.info(f"step {global_step+1}: eval_acc = {eval_similarity:.4f}")

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}.pt'.format(global_step, eval_similarity)), self.spk_emb_encoder)

                global_step += 1
                epoch_loss += cs_loss.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()

    def evaluate_spk_encoder(self, val_dataloader, global_step):
        cs_loss_fn = torch.nn.CosineEmbeddingLoss()
        similaritys = []
        cs_losses = []
        self.spk_emb_encoder.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                avhubert_feat = batch['avhubert_feat']
                avhubert_feat_len = batch['avhubert_feat_len']
                spk_emb = batch['spk_emb']
                spk_emb_hat = self.spk_emb_encoder(avhubert_feat, avhubert_feat_len)
                cs_loss = cs_loss_fn(spk_emb_hat, spk_emb, torch.ones(spk_emb_hat.size(0)))
                similarity = F.cosine_similarity(spk_emb_hat, spk_emb)
                similaritys.append(similarity)
                cs_losses.append(cs_loss.item())
        self.writer.add_scalar(f"eval_spk/similarity", np.mean(similaritys), global_step)
        self.writer.add_scalar(f"eval_spk/cs_loss", np.mean(cs_losses), global_step)
        self.spk_emb_encoder.train()
        return np.mean(similaritys)
