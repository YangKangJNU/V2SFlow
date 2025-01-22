## V2SFlow: Video-to-Speech Generation with  Speech Decomposition and Rectified Flow

Unofficial implementation of V2SFlow in Pytorch

## Usage:
```
## LJSpeech
python src/preprocess/extract_pitch.py
python train_vq_vae.py
## LRS3
python src/preprocess/extract_spk_emb.py
python train_encoder.py
python train_decoder.py
```

## Reference Repository
