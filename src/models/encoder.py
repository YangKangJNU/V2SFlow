from torch import nn
import torch

from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,  # noqa: H301
)

from espnet.nets.pytorch_backend.transformer.repeat import repeat

from src.models.attention import RelPositionMultiHeadedAttention

from src.utils import lens_to_mask


class Conformer(nn.Module):
    def __init__(self,
                input_dim=1024,
                num_layer=6,
                attention_dim=256,
                attention_head=4,
                kernel_size=31,
                feedforward_dim=2048,
                dropout_rate=0.1,
                ):
        super().__init__()

        pos_enc_class = RelPositionalEncoding
        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (attention_head, attention_dim, dropout_rate)
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, feedforward_dim, dropout_rate)
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, kernel_size)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attention_dim),
            pos_enc_class(attention_dim, dropout_rate),
        )
        self.encoder = repeat(
            num_layer,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args),
                dropout_rate,
                normalize_before=True,
                concat_after=True,
                macaron_style=True,                
            ),
        )

    def forward(self, x, x_len):
        x = self.embed(x)
        x_mask = lens_to_mask(x_len).unsqueeze(1)
        x, x_mask = self.encoder(x, x_mask)

        return x[0]*(x_mask.permute(0, 2, 1)), x_mask.squeeze()

class Encoder(nn.Module):
    def __init__(self, cfg=None,):
        super().__init__()
        self.num_layer = cfg.num_layer
        self.attention_dim = cfg.attention_dim
        self.attention_head = cfg.attention_head
        self.kernel_size = cfg.kernel_size
        self.feedforward_dim = cfg.feedforward_dim
        self.dropout_rate = cfg.dropout_rate
        self.num_tokens = cfg.num_tokens

        self.conformer = Conformer(num_layer=self.num_layer, 
                                   attention_dim=self.attention_dim, 
                                   attention_head=self.attention_head, 
                                   kernel_size=self.kernel_size, 
                                   feedforward_dim=self.feedforward_dim, 
                                   dropout_rate=self.dropout_rate)
        self.upsample = nn.ConvTranspose1d(in_channels=self.attention_dim, out_channels=self.attention_dim, kernel_size=2, stride=2)
        self.fc = nn.Linear(self.attention_dim, self.num_tokens)

    def forward(self, x, x_len):
        x, _ = self.conformer(x, x_len)
        x = self.upsample(x.permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.fc(x)
        return logits
    
class SpkEncoder(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.num_layer = cfg.num_layer
        self.attention_dim = cfg.attention_dim
        self.attention_head = cfg.attention_head
        self.kernel_size = cfg.kernel_size
        self.feedforward_dim = cfg.feedforward_dim
        self.dropout_rate = cfg.dropout_rate
        self.spk_emb_dim = cfg.spk_emb_dim
        self.conformer = Conformer(num_layer=self.num_layer, 
                                   attention_dim=self.attention_dim, 
                                   attention_head=self.attention_head, 
                                   kernel_size=self.kernel_size, 
                                   feedforward_dim=self.feedforward_dim, 
                                   dropout_rate=self.dropout_rate)
        self.fc = nn.Linear(self.attention_dim, self.spk_emb_dim)

    def forward(self, x, x_len):
        x, _ = self.conformer(x, x_len)
        spk_emb = self.fc(x)
        spk_emb_mean = torch.mean(spk_emb, dim=1)
        return spk_emb_mean