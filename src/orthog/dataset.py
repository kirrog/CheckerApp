from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.orthog.augmentation import AUGMENTATIONS
from src.orthog.pretrained import TOKEN_IDX


class BaseDataset(Dataset):

    def __init__(self,
                 files: str,
                 tokenizer: PreTrainedTokenizer,
                 sequence_len: int,
                 token_style: str,
                 *args,
                 **kwargs) -> None:
        self.seq_len = sequence_len
        self.token_style = token_style
        self.tokenizer = tokenizer
        self.data_x = np.load(str(Path(files) / "aug.npy"))
        self.data_y = np.load(str(Path(files) / "clear.npy"))
        assert self.data_x.shape[0] == self.data_y.shape[0]

    @classmethod
    def parse_tokens(cls,
                     tokens: str,
                     tokenizer: PreTrainedTokenizer,
                     seq_len: int,
                     token_style: str,
                     *args,
                     **kwargs) -> List[List[List[int]]]:
        """
        Convert tokenized data for model prediction

        Args:
            tokens (`Union[list[str], tuple[str]]`): splited tokens
            tokenizer (`PreTrainedTokenizer`): tokenizer which split tokens to subtokens
            seq_len (`int`): sequence length
            token_style (`str`): token_style from pretrained.TOKEN_IDX

        Returns:
            (`list[BatchWithoutTarget]`): list of bathces

        ```txt
        tokens    : [token  token  ##token  PAD ]
             x    : [321    1233   23121    101 ]
        attn_mask : [1      1      1        0   ]
        ```

        """
        data_items = []
        x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokens))

        if len(x) > seq_len - 2:
            for t in range(((len(x) - 1) // seq_len) + (1 if ((len(x) - 1) % seq_len > 0) else 0)):
                x = x[t * (seq_len - 2):(t + 1) * (seq_len - 2)]
                r = ([TOKEN_IDX[token_style]['START_SEQ']] + x +
                     [TOKEN_IDX[token_style]['UNK'] for _ in range(seq_len - len(x) - 2)] +
                     [TOKEN_IDX[token_style]['END_SEQ']])
                mask = [1 if token != TOKEN_IDX[token_style]['UNK'] else 0 for token in r]
                data_items.append([r, mask])
        else:
            r = ([TOKEN_IDX[token_style]['START_SEQ']] + x +
                 [TOKEN_IDX[token_style]['UNK'] for _ in range(seq_len - len(x) - 2)] +
                 [TOKEN_IDX[token_style]['END_SEQ']])
            mask = [1 if token != TOKEN_IDX[token_style]['UNK'] else 0 for token in r]
            data_items.append([r, mask])

        return data_items

    def __len__(self) -> int:
        return self.data_x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.data_x[index]
        attn_mask = np.zeros(x.shape)
        attn_mask[x != (TOKEN_IDX[self.token_style]['UNK'])] = 1
        y = self.data_y[index]
        y_mask = np.zeros(y.shape)
        y_mask[y != (TOKEN_IDX[self.token_style]['UNK'])] = 1

        x = torch.tensor(x)  # type: ignore
        attn_mask = torch.tensor(attn_mask)  # type: ignore
        y = torch.tensor(y)  # type: ignore
        y_mask = torch.tensor(y_mask)  # type: ignore

        return x, y, attn_mask, y_mask  # type: ignore


class ReorthDataset(BaseDataset):
    def __init__(self,
                 file: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 sequence_len: int,
                 token_style: str,
                 is_train=False,
                 augment_rate=0.,
                 augment_type='substitute',
                 *args,
                 **kwargs) -> None:
        """Preprocess data for restore punctuation

        Args:
            file (`str`): single file or list of text files containing tokens and punctuations separated by tab in lines
            sequence_len (`int`): length of each sequence
            token_style (`str`): For getting index of special tokens in pretrained.TOKEN_IDX
            is_train (`bool, optional`): if false do not apply augmentation. Defaults to False.
            augment_rate (`float, optional`): percent of data which should be augmented. Defaults to 0.0.
            augment_type (`str, optional`): augmentation type. Defaults to 'substitute'.
        """
        super().__init__(file, tokenizer, sequence_len, token_style, *args, **kwargs)

        self.is_train = is_train
        self.augment_type = augment_type
        self.augment_rate = augment_rate

    def _augment(self, x, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.seq_len:
            # len increased due to insert
            x_aug = x_aug[:self.seq_len]
            y_aug = y_aug[:self.seq_len]
            y_mask_aug = y_mask_aug[:self.seq_len]
        elif len(x_aug) < self.seq_len:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['UNK'] for _ in range(self.seq_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.seq_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.seq_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['UNK'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.data_x[index]
        attn_mask = np.zeros(x.shape)
        attn_mask[x != TOKEN_IDX[self.token_style]['UNK']] = 1
        y = self.data_y[index]
        y_mask = np.zeros(y.shape)
        y_mask[y != TOKEN_IDX[self.token_style]['UNK']] = 1

        # if self.is_train and self.augment_rate > 0:
        #     x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)  # type: ignore
        attn_mask = torch.tensor(attn_mask)  # type: ignore
        y = torch.tensor(y)  # type: ignore
        y_mask = torch.tensor(y_mask)  # type: ignore

        return x, y, attn_mask, y_mask  # type: ignore
