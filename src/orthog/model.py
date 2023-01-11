import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.orthog.pretrained import PRETRAINED_MODELS

Path_type = Union[Path, str, os.PathLike]


class CorrectionModel(nn.Module):
    def __init__(self,
                 pretrained_model: str,
                 freeze_pretrained: Optional[bool] = False,
                 *args,
                 **kwargs) -> None:

        super(CorrectionModel, self).__init__()

        self.pretrained_transformer = PRETRAINED_MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        self.hidden_size = PRETRAINED_MODELS[pretrained_model][2]

        if freeze_pretrained:
            for p in self.pretrained_transformer.parameters():
                p.requires_grad = False

    def forward(self, x: Tensor, attn_masks: Tensor) -> Tensor:
        # add dummy batch for single sample
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        # (B, N, E) -> (B, N, E)
        x = self.pretrained_transformer(x, attention_mask=attn_masks)[0]
        return x

    def save(self, save_path: Path_type) -> None:
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: Path_type, *args, **kwargs) -> None:
        self.load_state_dict(torch.load(load_path, *args, **kwargs))

    def quantize(self, *args, **kwargs):
        quantized_model = torch.quantization.quantize_dynamic(
            self, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
