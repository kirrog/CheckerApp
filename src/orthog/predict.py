from pathlib import Path
from typing import Dict, Optional, Type, Union

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.orthog.dataset import BaseDataset
from src.orthog.model import CorrectionModel
from src.orthog.pretrained import PRETRAINED_MODELS
from src.orthog.utils import get_last_pretrained_weight_path, load_params


class BasePredictor:
    def __init__(self,
                 model_name: str,
                 models_root: Path = Path("models"),
                 dataset_class: Type[BaseDataset] = BaseDataset,
                 model_weights: Optional[str] = None,
                 quantization: Optional[bool] = False,
                 *args,
                 **kwargs,
                 ) -> None:
        if model_name == "DeepPavlov/rubert-base-cased-sentence":
            self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            self.device = torch.device('cuda' if (not quantization) and torch.cuda.is_available() else 'cpu')
            # self.params = load_params(Path("models/reorth-model"))
            self.params = load_params(Path("/home/kirrog/projects/CheckerApp/models/reorth-model"))
            self.dataset_class = dataset_class
        else:
            model_dir = models_root / model_name
            self.params = load_params(model_dir)
            self.device = torch.device('cuda' if (not quantization) and torch.cuda.is_available() else 'cpu')

            if not model_weights:
                self.weights = get_last_pretrained_weight_path(model_dir)
            else:
                self.weights = model_dir / 'weights' / model_weights

            self.model = self.load_model(quantization=quantization)
            self.tokenizer = self.load_tokenizer()
            self.dataset_class = dataset_class

    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:
        model = CorrectionModel(self.params['pretrained_model'],
                                self.params['freeze_pretrained'])

        if quantization:
            model = model.quantize()

        model.to(self.device)
        model.load(self.weights, map_location=self.device)
        model.eval()
        return model

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        name = self.params['pretrained_model']
        tokenizer = PRETRAINED_MODELS[name][1].from_pretrained(name)
        return tokenizer


class ReorthPredictor(BasePredictor):
    def __call__(self, text: str) -> str:
        token_style = PRETRAINED_MODELS[self.params['pretrained_model']][3]
        seq_len = self.params['sequence_length']

        data = torch.tensor(self.dataset_class.parse_tokens(text,
                                                            self.tokenizer,
                                                            seq_len,
                                                            token_style))

        with torch.no_grad():
            r = []
            for case in data:
                y_predict = self.model(case[0], case[1])
                t = self.dataset_class.tokenizer.decode(y_predict)
                r.append(t)
            result = " ".join(r)

        result = result.strip()
        return result
