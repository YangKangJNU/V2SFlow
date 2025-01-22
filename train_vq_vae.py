
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
from src.utils import build_env, AttrDict
from src.vq_vae_pytorch.vq_vae import Quantizer
from src.trainer_vq_vae import VQVAETrainer, F0Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="data/LJSpeech-1.1")
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_workers', default=64)
    parser.add_argument('--output_dir', default='logs/f0_vqvae')
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--log_step', default=100)
    parser.add_argument('--val_step', default=500)
    parser.add_argument('--save_step', default=1000)
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--config', default='config/f0_vqvae.json')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(args.config, 'config.json', args.output_dir)

    generator = Quantizer(h)

    train_dataset = F0Dataset(args.root, 'train')
    val_dataset = F0Dataset(args.root, 'val')


    trainer = VQVAETrainer(
        generator,
        lr=args.lr,
        grad_accumulation_steps = 1,
        tensorboard_log_dir=args.output_dir,
        log_file = os.path.join(args.output_dir, 'f0_vqvae.txt'),
        adam_b1=h['adam_b1'],
        adam_b2=h['adam_b2'],
        lr_decay=h['lr_decay']
    )


    trainer.train(train_dataset,
                   args.epochs,
                   args.batch_size,
                   args.num_workers, 
                   val_dataset=val_dataset,
                   log_step=args.log_step, 
                   save_step=args.save_step, 
                   val_step=args.val_step,
                   ckpt_path=args.ckpt)

if __name__ == '__main__':
    main()
