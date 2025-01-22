
import argparse
import json
import os

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from src.trainer_encoder import (
    LRS3Dataset,
    Trainer,
)
import hydra
from omegaconf import DictConfig
from src.models.encoder import Encoder, SpkEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="")
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_workers', default=10)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--log_step', default=500)
    parser.add_argument('--val_step', default=2000)
    parser.add_argument('--save_step', default=4000)
    parser.add_argument('--ckpt', default='')
    args = parser.parse_args()
    return args

@hydra.main(config_path="configs/encoder", config_name="default.yaml")
def main(cfg: DictConfig):
    args = parse_args()
    train_dataset = LRS3Dataset(args.root, 'train')
    val_dataset = LRS3Dataset(args.root, 'val')

    content_encoder = Encoder(cfg.content)
    pitch_encoder = Encoder(cfg.pitch)
    spk_encoder = SpkEncoder(cfg.spk)
    trainer = Trainer(
        content_encoder=content_encoder,
        pitch_encoder=pitch_encoder,
        spk_emb_encoder=spk_encoder,
        num_warmup_steps=10000,
        lr=args.lr,
        grad_accumulation_steps = 1,
        tensorboard_log_dir=args.output_dir,
        checkpoint_path = os.path.join(args.output_dir, 'model.pt'),
        log_file = os.path.join(args.output_dir, 'logs.txt')
    )

    trainer.train_content_encoder(train_dataset,
                   args.epochs, 
                   args.batch_size, 
                   val_dataset = val_dataset,
                   num_workers=args.num_workers, 
                   log_step=args.log_step, 
                   save_step=args.save_step, 
                   val_step=args.val_step,
                   ckpt_path=args.ckpt)

if __name__ == '__main__':
    main()
