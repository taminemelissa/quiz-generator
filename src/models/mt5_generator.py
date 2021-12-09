import pytorch_lightning as pl
import os
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from src.data.data_format import *
from src.data.utils import yield_batches
from torch.nn import DataParallel


class MT5Generator(pl.LightningModule):
    def __init__(self, model_path: str):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)

    def generate(self, questions: List[Question]) -> List[Question]:
        result = []
        all_answers = QuestionContextAnswer(questions=questions).get_all_answers()
        if all_answers:
            model = self.model.module if isinstance(self.model, DataParallel) else self.model
            for batch in yield_batches(all_answers, batch_size=len(questions)):
                answers = [a.text for a in batch]
                contexts = [a.context.text for a in batch]
                source_encoding = self.tokenizer(answers,
                                                 contexts,
                                                 max_length=512,
                                                 padding='max_length',
                                                 truncation=True,
                                                 return_attention_mask=True,
                                                 add_special_tokens=True,
                                                 return_tensors='pt')
                source_encoding = source_encoding.to(self.device)
                generated_ids = model.generate(input_ids=source_encoding['input_ids'],
                                               attention_mask=source_encoding['attention_mask'],
                                               num_beams=4,
                                               max_length=32,
                                               repetition_penalty=2.0,
                                               length_penalty=1.0,
                                               early_stopping=True,
                                               use_cache=True)
                generated_questions = self.tokenizer.batch_decode(generated_ids,
                                                                  skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=True)
                generated_items = [Question(text=q, predicted_answers=[batch[i]])
                                   for i, q in enumerate(generated_questions)]
                result += generated_items

        return result
