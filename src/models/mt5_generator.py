import pytorch_lightning as pl
from typing import Union
import os
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


class MT5Generator(pl.LightningModule):
    def __init__(self, config: Union[dict, str] = None):
        super().__init__()
        model_path = config['model']
        if os.path.isdir(model_path):
            self.save_hyperparameters()
            self.config = config
            print(self.hparams)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
            self.tokenizer = MT5Tokenizer.from_pretrained(model_path)
        else:
            raise Exception('Path in config is incorrect')