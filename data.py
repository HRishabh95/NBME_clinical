import pandas as pd
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
# Seeds all the processes including numpy torch and other imported modules.
# check the version provided by Lightning
import pytorch_lightning as pl
from torch.utils.data import Dataset


class Manu_Dataset(Dataset):
   def __init__(
       self,
       data:pd.DataFrame,
       tokenizer:T5Tokenizer,
       source_max_token_len: int = 396,
       target_max_token_len: int = 32,
       ):
     self.data =  data
     self.tokenizer =  tokenizer
     self.source_max_token_len =  source_max_token_len
     self.target_max_token_len =  target_max_token_len

   def __len__(self):
     return len(self.data)

   def __getitem__(self, index: int):
     data_row = self.data.iloc[index]
     source_encoding = self.tokenizer(
       data_row['question'],
       data_row['context'],
       max_length=self.source_max_token_len,
       padding='max_length',
       truncation="only_second",
       return_attention_mask=True,
       add_special_tokens=True,
       return_tensors="pt"
       )
     target_encoding = self.tokenizer(
       data_row['answer_text'],
       max_length=self.target_max_token_len,
       padding='max_length',
       truncation=True,
       return_attention_mask=True,
       add_special_tokens=True,
       return_tensors="pt"
       )
     labels = target_encoding['input_ids']
     labels[labels==0] = -100
     return dict(
         question=data_row['question'],
         context=data_row['context'],
         answer_text=data_row['answer_text'],
         input_ids=source_encoding["input_ids"].flatten(),
         attention_mask=source_encoding['attention_mask'].flatten(),
         labels=labels.flatten()
     )
