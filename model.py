from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import json
import pytorch_lightning as pl

MODEL_NAME ='t5-base'

class QAModel(pl.LightningModule):
   def __init__(self):
     super().__init__()
     self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

   def forward(self, input_ids, attention_mask, labels=None):
     output = self.model(
         input_ids,
         attention_mask=attention_mask,
         labels=labels)
     return output.loss, output.logits

   def training_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("train_loss", loss, prog_bar=True, logger=True)
     return {"loss": loss, "predictions":outputs, "labels": labels}

   def validation_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("val_loss", loss, prog_bar=True, logger=True)
     return loss

   def test_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("test_loss", loss, prog_bar=True, logger=True)
     return loss,outputs



   def test_end(self,outputs):
       f=open('text.lst','w')
       f.write(json.dumps(outputs))
       f.close()
       return outputs

   def configure_optimizers(self):
     optimizer = AdamW(self.parameters(), lr=0.0001)
     return optimizer