
from random import random
from torch import nn
import torch


from src.rfm_decoder_pytorch.modules import TimestepEmbedding, ConvPositionEmbedding, DiTBlock, AdaLayerNormZero_Final
from x_transformers.x_transformers import RotaryEmbedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, content_dim, pitch_dim, spk_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + content_dim + pitch_dim + spk_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, content_emb, pitch_emb, spk_emb, drop_cond=False):
        T = x.shape[1]
        spk_emb = spk_emb.unsqueeze(1).repeat(1, T, 1)
        cond = torch.cat([content_emb, pitch_emb, spk_emb], dim=-1)
        if drop_cond:
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat([x, cond], dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

class DiT(nn.Module):
    def __init__(self,
                 cfg,):
        super().__init__()
        self.dim = cfg.dim
        self.depth = cfg.depth
        self.heads = cfg.heads
        self.dim_head = cfg.dim_head
        self.dropout = cfg.dropout
        self.ff_mult = cfg.ff_mult
        self.mel_dim = cfg.mel_dim
        self.content_nums = cfg.content_nums
        self.content_dim = cfg.content_dim
        self.pitch_nums = cfg.pitch_nums
        self.pitch_dim = cfg.pitch_dim
        self.spk_dim = cfg.spk_dim
        self.long_skip_connection = cfg.elf.long_skip_connection
        self.time_embed = TimestepEmbedding(self.dim)

        self.content_embed = nn.Embedding(self.content_nums+1, self.content_dim)
        self.pitch_embed = nn.Embedding(self.pitch_nums+1, self.pitch_dim)

        self.input_embed = InputEmbedding(self.mel_dim, self.content_dim, self.pitch_dim, self.spk_dim, self.dim)

        self.rotary_embed = RotaryEmbedding(self.dim_head)
        self.long_skip_connection = nn.Linear(self.dim * 2, self.dim, bias=False) if self.long_skip_connection else None

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=self.dim, heads=self.heads, dim_head=self.dim_head, ff_mult=self.ff_mult, dropout=self.dropout) for _ in range(self.depth)]
        )

        self.norm_out = AdaLayerNormZero_Final(self.dim)
        self.proj_out = nn.Linear(self.dim, self.mel_dim)


    def forward(self,
                x,
                time,
                content,
                pitch,
                spk,
                drop_cond=False,
                mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        content_emb = self.content_embed(content)
        pitch_emb = self.pitch_embed(pitch)
        x = self.input_embed(x, content_emb, pitch_emb, spk, drop_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output*mask.unsqueeze(-1)