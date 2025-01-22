import os
import einops
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils import default, denormalise_mel, exists
from adam_atan2_pytorch.adopt import Adopt
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import torch.nn.functional as F
from src.utils import STFT, load_wav_to_torch, normalise_mel, plot_spectrogram
from ema_pytorch import EMA


class LRS3Dataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.filelist, self.f_names = self.get_filelist(self.root, self.mode)
        self.hubert_unit_dict = self.build_hubert_unit_dict()
        self.pitch_unit_dict = self.build_pitch_unit_dict()
        self.stft = STFT(filter_length=1024, hop_length=160, win_length=640, sampling_rate=16000, mel_fmin=55., mel_fmax=7500.)

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
    
    def get_mel(self, filename):
        audio, _ = load_wav_to_torch(filename)
        audio = audio / 1.1 / audio.abs().max()
        melspectrogram = self.stft.get_mel(audio)
        return audio, melspectrogram

    def __getitem__(self, index):
        file_path = self.filelist[index]
        f_name = self.f_names[index]
        spk_emb_path = file_path.replace('/video/', '/spk_emb/') + '.pt'
        spk_emb = torch.load(spk_emb_path).squeeze(0)
        hubert_unit = torch.tensor(self.hubert_unit_dict[f_name]) + 1
        pitch_unit = torch.tensor(self.pitch_unit_dict[f_name]) + 1
        audio_path = file_path.replace('/video/', '/audio/') + '.wav'
        audio, mel = self.get_mel(audio_path)
        mel = normalise_mel(mel)
        if mel.size(1) % 2 != 0:
            padding_mel = torch.zeros(mel.size(0), 1, dtype=mel.dtype)
            mel = torch.cat((mel, padding_mel), dim=1)
        mel = einops.rearrange(mel, 'c (t n) -> (c n) t', n=2)
        audio = torch.FloatTensor(audio)

        diff = len(hubert_unit) - len(pitch_unit)
        if diff < 0:
            padding = torch.zeros(-diff, dtype=pitch_unit.dtype)
            pitch_unit = torch.cat((pitch_unit, padding))
        elif diff > 0:
            pitch_unit = pitch_unit[:-diff]

        diff = len(hubert_unit) - len(audio)*320
        if diff < 0:
            padding_audio = torch.zeros(-diff, dtype=audio.dtype)
            audio = torch.cat((audio, padding_audio), dim=0)
        elif diff > 0:
            audio = audio[:-diff]

        diff = len(hubert_unit) - mel.size(1)
        if diff < 0:
            padding_mel = torch.zeros(mel.size(0), -diff, dtype=mel.dtype)
            mel = torch.cat((mel, padding_mel), dim=1)
        elif diff > 0:
            mel = mel[:, :-diff]

        return spk_emb, hubert_unit, pitch_unit, mel, audio, f_name


def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(0) for x in batch]),
        dim=0, descending=True)
    
    max_hubert_len = max([x[1].size(0) for x in batch])
    max_pitch_len = max([x[2].size(0) for x in batch])
    max_mel_len = max([x[3].size(1) for x in batch])
    max_audio_len = max([x[4].size(0) for x in batch])

    hubert_lengths = torch.IntTensor(len(batch))
    pitch_lengths = torch.IntTensor(len(batch))
    mel_lengths = torch.IntTensor(len(batch))
    audio_lengths = torch.IntTensor(len(batch))

    hubert_padded = torch.zeros(len(batch), max_hubert_len, dtype=torch.int64)
    pitch_padded = torch.zeros(len(batch), max_pitch_len, dtype=torch.int64)
    mel_padded = torch.zeros(len(batch), 160, max_mel_len, dtype=torch.float32)
    audio_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
    
    hubert_padded.zero_()
    pitch_padded.zero_()
    mel_padded.zero_()
    audio_padded.zero_()

    spk_emb = []
    file_name = []

    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        spk_emb.append(row[0])

        hubert_unit = row[1]
        hubert_padded[i, :row[1].size(0)] = hubert_unit
        hubert_lengths[i] = row[1].size(0)

        pitch = row[2]
        pitch_padded[i, :row[2].size(0)] = pitch
        pitch_lengths[i] = row[2].size(0)

        mel = row[3]
        mel_padded[i, :, :row[3].size(1)] = mel
        mel_lengths[i] = row[3].size(1)

        audio = row[4]
        audio_padded[i, :row[4].size(0)] = audio
        audio_lengths[i] = row[4].size(0)

        file_name.append(row[5])
    
    spk_emb = torch.stack(spk_emb)

    return dict(
        hubert_unit = hubert_padded,
        hubert_unit_len = hubert_lengths,
        pitch_unit = pitch_padded,
        pitch_unit_len = pitch_lengths,
        spk_emb = spk_emb,
        mel = mel_padded,
        mel_len= mel_lengths,
        audio = audio_padded,
        audio_len = audio_lengths,
        file_name = file_name
    )



class Trainer:
    def __init__(
            self,
            model = None,
            vocoder = None,
            lr = 1e-4,
            checkpoint_path = None,
            grad_accumulation_steps = 1,
            log_file = 'logs.txt',
            tensorboard_log_dir = 'logs/test1',
            accelerate_kwargs: dict = dict(),
            ema_kwargs: dict = dict(),
            num_warmup_steps = None,
            sample_steps = 50,
            max_grad_norm = 1.0,
            use_switch_ema = False,
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
        self.model = model
        self.vocoder = vocoder
        self.lr = lr
        self.ema_model = EMA(
            model,
            include_online_model = False,
            **ema_kwargs
        )
        self.use_switch_ema = use_switch_ema 
        if not exists(optimizer):
            optimizer = Adopt(model.parameters(), lr = lr)
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = default(checkpoint_path, 'model.pth') 
        self.sample_steps = sample_steps
        self.tensorboard_log_dir = tensorboard_log_dir
        self.model, self.ema_model, self.vocoder, self.optimizer = self.accelerator.prepare(
            self.model, self.ema_model, self.vocoder, self.optimizer
        )
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.scheduler = None

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def save_checkpoint(self, step, ckpt_name, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict = self.ema_model.state_dict(),
                scheduler_state_dict = self.scheduler.state_dict(),
                step = step
            )

            self.accelerator.save(checkpoint, ckpt_name)


    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path) or not os.path.exists(ckpt_path):
            return 0

        checkpoint = torch.load(ckpt_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['step']

    def train(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, save_step=2000, val_step=1000):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(train_dataloader, val_dataloder, self.scheduler)
        start_step = self.load_checkpoint(ckpt_path)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1

        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.content_encoder):            
                    hubert_unit = batch['hubert_unit']
                    pitch_unit = batch['pitch_unit']
                    spk_emb = batch['spk_emb']
                    mel = batch['mel']
                    mel_len = batch['mel_len']
                    audio = batch['audio']
                    loss = self.model(mel, hubert_unit, pitch_unit, spk_emb, mel_len)
                    self.accelerator.backward(loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:        
                        logger.info(f"step {global_step+1}: loss = {loss.item():.4f}")
                        self.writer.add_scalar('train_content/loss', loss.item(), global_step)

                if global_step % val_step == 0:
                    eval_err = self.evaluate(val_dataloder, global_step)
                    logger.info(f"step {global_step+1}: eval_err = {eval_err:.4f}s")

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}.pt'.format(global_step, eval_err)))

                global_step += 1
                epoch_loss += loss.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()

    def evaluate(self, val_dataloader, global_step, n_timesteps=30):
        mel_loss_fn = torch.nn.L1Loss()
        mel_errs = []
        self.model.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                hubert_unit = batch['hubert_unit']
                hubert_unit_len = batch['hubert_unit_len']
                pitch_unit = batch['pitch_unit']
                spk_emb = batch['spk_emb']
                mel = batch['mel']
                audio = batch['audio']
                mel_hat, _ = self.model.sample(hubert_unit, pitch_unit, spk_emb, hubert_unit_len, n_timesteps)
                mel_hat = denormalise_mel(einops.rearrange(mel_hat, 'b (c n) t -> b c (t n)', n = 2))
                mel = denormalise_mel(einops.rearrange(mel,'b (c n) t -> b c (t n)', n = 2))
                mel_err = mel_loss_fn(mel_hat, mel)
                mel_errs.append(mel_err.item())
        self.writer.add_scalar(f"eval/mel_errs", np.mean(mel_errs), global_step)
        if self.vocoder is not None:
            y_hat = self.vocoder(mel_hat).squeeze()
            target_lens = hubert_unit_len*320
            for _ in range(hubert_unit.size(0)):
                self.writer.add_audio(f"audio_{_}/gen", y_hat[_, :target_lens[_]], global_step, sample_rate=16000)
                self.writer.add_audio(f"audio_{_}/target", audio[_, :target_lens[_]], global_step, sample_rate=16000)
        for _ in range(hubert_unit.size(0)):
            mel_lens = hubert_unit_len*2
            self.writer.add_figure(f"mel_{_}/pred", plot_spectrogram(mel_hat.permute(0, 2, 1)[_, :mel_lens[_], :]), global_step)
            self.writer.add_figure(f"mel_{_}/gt", plot_spectrogram(mel.permute(0, 2, 1)[_, :mel_lens[_], :]), global_step)                       
        self.model.train()
        return np.mean(mel_errs)       
                    
