import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
from src.preprocess.audio import preprocess_wav, wav_to_mel_spectrogram
from glob import glob
from tqdm import tqdm
import json
from src.utils import build_env, AttrDict
from src.vq_vae_pytorch.vq_vae import Quantizer
from src.trainer_vq_vae import load_audio, normalize, get_yaapt_f0, MAX_WAV_VALUE

def get_emb(filepath, model):
    audio = preprocess_wav(filepath)
    frames = wav_to_mel_spectrogram(audio)
    frames = torch.from_numpy(frames).cuda()
    with torch.no_grad():
        embed = model.forward(frames.unsqueeze(0))
    emb_filepath = filepath.replace('/audio/', '/spk_emb/').replace('.wav', '.pt')
    os.makedirs(os.path.dirname(emb_filepath), exist_ok=True)
    torch.save(embed, emb_filepath)


def get_filelist(dirpath):
    filelist = glob(os.path.join(dirpath, 'audio', '*', '*', '*.wav'))
    return filelist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--config', type=str, default='configs/f0_vqvae.json')
    parser.add_argument('--output', type=str, default='data/LRS3/pitch.csv')
    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Quantizer(h)
    ckpt = torch.load(args.ckpt)
    generator.load_state_dict(ckpt['model_state_dict'])
    filelist = get_filelist(args.root)
    for file in tqdm(filelist):
        audio_file = file
        audio, sr = load_audio(file)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        f0_path = file.replace('/audio/', '/f0_yaapt/').replace('.wav', '.npy')
        if os.path.exists(f0_path):
            f0 = np.load(f0_path).astype(np.float32)
        else:
            os.makedirs(os.path.dirname(f0_path), exist_ok=True)
            f0 = get_yaapt_f0(audio.numpy(), rate=16000, interp=False)
            f0 = f0.astype(np.float32)
            np.save(f0_path, f0)
            print(f'f0 have been saved in {f0_path}')
        ii = f0 != 0
        f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
        f0 = torch.FloatTensor(f0)
        code = generator.get_code(f0).tolist()[0]
        code_str = ' '.join([str(i) for i in code])
        f_name = '/'.join(file.split('/')[-3:]).split('.')[0]
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'a') as f:
            f.write(f_name + '|' + code_str + '\n')










