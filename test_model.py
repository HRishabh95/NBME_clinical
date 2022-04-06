import pandas as pd
from pathlib import Path
import re
from transformers import T5TokenizerFast
from model import QAModel


nbme=pd.read_csv('./NBME_test.csv',index_col=0)

trained_model = QAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
trained_model.freeze()

MODEL_NAME ='t5-base'
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

def generate_answer(question):
  source_encoding=tokenizer(
      question["question"],
      question['context'],
      max_length = 396,
      padding="max_length",
      truncation="only_second",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt",
      return_offsets_mapping=True
  )
  generated_ids = trained_model.model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      num_beams=1,  # greedy search
      max_length=80,
      repetition_penalty=2.5,
      early_stopping=True,
      use_cache=True)
  preds = [
          tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]
  return "".join(preds)


nbme_test=pd.read_csv('./NBME_test.csv',index_col=0)
ids=['00016_000','00016_001','00016_002','00016_003','00016_004']
for ii,rows in nbme_test.iterrows():
    nbme_test.at[ii,'ids']=ids[ii]


location=[]
for ii,rows in nbme_test.iterrows():
    sample=rows
    ans=generate_answer(sample)
    ans_words=ans.split()
    ans_len=len(ans)
    spos=[m.start() for m in re.finditer(ans, sample.context)]
    loca=''
    if len(spos)==0:
        start=ans_words[0]
        spos = [m.start() for m in re.finditer(start, sample.context)]
        end = ans_words[-1]
        epos = [m.start() for m in re.finditer(end, sample.context)]
        for ii in spos:
            for jj in epos:
                if jj>ii:
                    if ans_len-5<=jj-ii<=ans_len+5:
                        loca+=f'''{ii} {jj};'''
    else:
        for spo in spos:
            loca+=f'''{spo} {int(spo)+ans_len};'''
    cleaned_loca=';'.join([i for i in loca.split(';') if len(i)>0])
    location.append([rows['ids'],cleaned_loca])


submission=pd.DataFrame(location,columns=['id','location'],index=None)

submission.to_csv('./submission_first.csv',index=None)


def loc_list_to_ints(loc_list):
    to_return = []
    for loc_str in loc_list:
        loc_strs = loc_str.split(";")
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
    return to_return



