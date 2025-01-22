







import os
import amfm_decompy.basic_tools as basic
import numpy as np
import amfm_decompy.pYAAPT as pYAAPT
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from librosa.util import normalize
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from src.utils import default, exists, plot_f0
from torch.utils.tensorboard import SummaryWriter
from adam_atan2_pytorch.adopt import Adopt
import torch.nn.functional as F
MAX_WAV_VALUE = 32768.0


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0

def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate

class F0Dataset(Dataset):
    def __init__(self, root, mode, sampling_rate=16000):
        self.root = root
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.filelist, self.filenames = self.get_filelist(self.root, self.mode)


    def get_filelist(self, root, mode):
        filelist = []
        f_names = []
        with open(os.path.join(root, f'{mode}.csv'), 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip()
                f_name = os.path.basename(file.split('/')[-1])
                filelist.append(file)
                f_names.append(f_name)
        return filelist, f_names

    def __getitem__(self, index):
        filename = self.filelist[index]
        audio, sr = load_audio(filename)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        f0_path = filename.replace('/wavs/', '/f0_yaapt/').replace('.wav', '.npy')
        if os.path.exists(f0_path):
            f0 = np.load(f0_path).astype(np.float32)
        else:
            os.makedirs(os.path.dirname(f0_path), exist_ok=True)
            f0 = get_yaapt_f0(audio.numpy(), rate=self.sampling_rate, interp=False)
            f0 = f0.astype(np.float32)
            np.save(f0_path, f0)
            print(f'f0 have been saved in {f0_path}')


        ii = f0 != 0
        f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
        f0 = torch.FloatTensor(f0)

        return f0.squeeze(), filename
    
    def __len__(self):
        return len(self.filelist)
    
def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(0) for x in batch]),
        dim=0, descending=True)
    
    max_f0_len = max([x[0].size(0) for x in batch])

    f0_lengths = torch.IntTensor(len(batch))

    f0_padded = torch.zeros(len(batch), max_f0_len, dtype=torch.float32)

    f0_padded.zero_()

    file_name = []

    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        f0 = row[0]
        f0_padded[i, :row[0].size(0)] = f0
        f0_lengths[i] = row[0].size(0)

        file_name.append(row[1])

    return dict(
        f0 = f0_padded.unsqueeze(1),
        f0_lengths = f0_lengths,
        file_name = file_name
    )
    

class VQVAETrainer:
    def __init__(
            self,
            model,
            lr = 1e-4,
            grad_accumulation_steps = 1,
            ckpt = None,
            log_file = 'logs.txt',
            tensorboard_log_dir = 'logs/test1',
            accelerate_kwargs: dict = dict(),
            adam_b1 = 0.8,
            adam_b2 = 0.99,
            lr_decay = 0.999,
            max_grad_norm = 1.0,
            sample_steps = 20,
    ):
        logger.add(log_file)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            # mixed_precision='fp16',
            **accelerate_kwargs
        )  
        self.model = model
        self.lr = lr
        self.model = self.accelerator.prepare(self.model)

        self.checkpoint_path = default(ckpt, 'model.pth')
        self.tensorboard_log_dir = tensorboard_log_dir
        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        self.lr_decay = lr_decay
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.scheduler = None
        self.optimizer = None
        self.max_grad_norm = max_grad_norm
        self.sample_steps = sample_steps
        
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, epoch, step, ckpt_name, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                epoch = epoch,
                step = step
            )

            self.accelerator.save(checkpoint, ckpt_name)

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path) or not os.path.exists(ckpt_path):
            return 0, 0

        checkpoint = torch.load(ckpt_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        if exists(self.optimizer):
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['step']
    
    def train(self, train_dataset, epochs, batch_size, num_workers, val_dataset=None, ckpt_path=None, log_step=100, save_step=2000, val_step=1000):
        self.optimizer = Adopt(self.model.parameters(), lr = self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, betas=[self.adam_b1, self.adam_b2])
        last_epoch, start_step = self.load_checkpoint(ckpt_path)
        global_step = start_step
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay, last_epoch=last_epoch-1)

        train_dataloader, val_dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(train_dataloader, val_dataloader, self.optimizer, self.scheduler)
        for epoch in range(last_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    f0 = batch['f0']
                    f0_lengths = batch['f0_lengths']
                    f_name = batch['file_name']

                    f0_g_hat, commit_loss, metrics = self.model(f0)
                    f0_commit_loss = commit_loss[0]
                    f0_metrics = metrics[0]

                    loss_recon = F.mse_loss(f0_g_hat, f0)
                    loss_recon += f0_commit_loss * 0.02

                    self.accelerator.backward(loss_recon)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if global_step % log_step == 0:
                    if self.accelerator.is_local_main_process:
                        logger.info(f"step {global_step}: loss = {loss_recon.item():.4f}")
                        self.writer.add_scalar('train/total_loss', loss_recon.item(), global_step)
                        self.writer.add_scalar('train/commit_error', f0_metrics['used_curr'].item(), global_step)
                        self.writer.add_scalar('train/entropy', f0_metrics['entropy'].item(), global_step)
                        self.writer.add_scalar('train/usage', f0_metrics['usage'].item(), global_step)
                        self.writer.add_image("train/f0", plot_f0(f0_g_hat[0].cpu().detach().numpy(), f0[0].cpu().numpy()), global_step, dataformats='HWC')
                
                if global_step % val_step == 0:
                    eval_err = self.evaluate(val_dataloader, self.sample_steps, global_step)
                    logger.info(f"step {global_step+1}: eval_wer = {eval_err:.4f}")

                if global_step % save_step == 0:
                    self.save_checkpoint(epoch, global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}.pt'.format(global_step,eval_err)))

                global_step += 1
                epoch_loss += loss_recon.item()
            
            self.scheduler.step()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()

    def evaluate(self, val_dataloader, sample_steps, global_step):
        eval_errs = []
        eval_f0_commit_losses = []
        self.model.eval()
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                f0 = batch['f0']
                f0_lengths = batch['f0_lengths']
                f_name = batch['file_name']

                f0_g_hat, commit_loss, _ = self.model(f0)
                f0_commit_loss = commit_loss[0]
                eval_f0_commit_losses.append(f0_commit_loss.item())

                eval_err = F.mse_loss(f0_g_hat, f0)
                eval_errs.append(eval_err.item())
        eval_errs_mean = np.mean(eval_errs).item()
        eval_f0_commit_losses_mean = np.mean(eval_f0_commit_losses).item()
        self.writer.add_scalar(f"eval/eval_err", eval_errs_mean, global_step)
        self.writer.add_scalar(f"eval/f0_commit_loss", eval_f0_commit_losses_mean, global_step)
        for _ in range(f0_g_hat.size(0)):
            self.writer.add_image(f"eval/f0_{_}", plot_f0(f0_g_hat[0, :, :f0_lengths[_]].cpu().detach().numpy(), f0[0, :, :f0_lengths[_]].cpu().numpy()), global_step, dataformats='HWC')
            self.writer.add_text(f"eval/audio_{_}", f_name[_], global_step)

        self.model.train()
        return eval_errs_mean




    
                           
                            





