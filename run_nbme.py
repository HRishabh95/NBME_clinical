import pandas as pd
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import DataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from termcolor import colored
# wrapping paragraphs into string
import textwrap
# model checkpoints in pretrained model
from pytorch_lightning.callbacks import ModelCheckpoint
from data import Dataset
from transformers import T5Tokenizer
from model import QAModel

nbme=pd.read_csv('./NBME_train.csv',index_col=0)

## Sample question
sample_question = nbme.iloc[243]
# Using textcolor to visualize the answer within the context
def color_answer(question):
    answer_start, answer_end = question["answer_start"],question["answer_end"]
    context = question['context']
    return  colored(context[:answer_start], "white") + \
    colored(context[answer_start:answer_end + 1], "green") + \
    colored(context[answer_end+1:], "white")

print(sample_question['question'])
print("Answer: ")
for wrap in textwrap.wrap(color_answer(sample_question), width = 100):
    print(wrap)



## Read Dataset
MODEL_NAME ='t5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
train_df, val_df = train_test_split(nbme, test_size=0.05)


BATCH_SIZE = 16
N_EPOCHS = 1
data_module = DataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

## Model
model = QAModel()

# To record the best performing model using checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints_nbme_b4_e6",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("training-logs", name="qa")

trainer = pl.Trainer(
    logger = logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate = 30
)

#trainer.fit(model, data_module)
#f=trainer.test()


# loading model
trained_model = QAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
f=trainer.test(trained_model,data_module.test_dataloader())
#trained_model.freeze()


## test the model witth loccactin prediction


