from random import random
from torch import nn
import torch
from src.utils import lens_to_mask
import torch.nn.functional as F
from torchdiffeq import odeint

class CFM(nn.Module):
    def __init__(self,
                 transformer,
                 sigma=0.0,
                 drop_prob=0.1,
                 odeint_kwargs: dict = dict(
                    # atol = 1e-5,
                    # rtol = 1e-5,
                    method="euler"
        )
                 ):
        super().__init__()

        self.sigma = sigma
        self.drop_prob = drop_prob
        self.transformer = transformer

        self.odeint_kwargs = odeint_kwargs

    @torch.no_grad()
    def sample(self,
               content,
               pitch,
               spk,
               lens,
               steps=30,
               guidance_scale=2.0,
               ):
        batch, seq_len, device = *content.shape[:2], content.device
        mask = lens_to_mask(lens, length=seq_len)

        def fn(t, x):
            pred = self.transformer(
                x = x, time=t, content=content, pitch=pitch, spk=spk, drop_cond=False, mask=mask
            )
            
            null_pred = self.transformer(
                x = x, time=t, content=content, pitch=pitch, spk=spk, drop_cond=True, mask=mask
            )

            return pred + (1 - guidance_scale) * null_pred
        
        y0 = torch.randn(batch, seq_len, 160)
        t_start = 0
        t = torch.linspace(t_start, 1, steps + 1, device=device)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]
        out = sampled
        return out, trajectory


    def forward(self,
                input,
                content,
                pitch,
                spk,
                lens=None):
        batch, seq_len, dtype, device, simga = *input.shape[:2], input.dtype, input.device, self.sigma

        mask = lens_to_mask(lens, length=seq_len)

        x1 = input
        x0 = torch.randn_like(x1)
        
        time = torch.rand((batch, ), dtype=dtype, device=device)
        t = torch.sigmoid(time)
        t = t.unsqueeze(-1).unsqueeze(-1)
        xt = (1 - t) * x0 + t * x1
        flow = x1 - x0

        drop_cond = random() < self.drop_prob

        pred = self.transformer(
            x = xt, time=time, content=content, pitch=pitch, spk=spk, drop_cond=drop_cond, mask=mask
        )

        loss = F.mse_loss(pred, flow, reduction='none')
        
        return loss.mean()
        
