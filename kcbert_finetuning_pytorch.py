from datasets import load_dataset
import torch
import transformers
import pandas as pd
#dataset = load_dataset("all.tsv",) #original
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
#raw_dataset=load_dataset('all.tsv')
#raw_dataset[100]
class_names=['0','1','2','3','4','5','6','7']
#class_names=[0,1,2,3,4,5,6,7]
from datasets import ClassLabel
from datasets import Value
from datasets import Features
dataset=load_dataset('csv',data_files={'train':'C:/Users/hyeyoung/PycharmProjects/test/kcbert/train_0323_v3.csv','test':'C:/Users/hyeyoung/PycharmProjects/test/kcbert/test_0323_v3.csv'})
#emotion_features=Features({'text':Value('string'),'label':ClassLabel(names=class_names)})
#dataset = load_dataset('csv', data_files='C:/Users/hyeyoung/PycharmProjects/test/kcbert/all0323.csv', delimiter=';', column_names=['label', 'text'], features=emotion_features)
#dataset=load_dataset('csv',data_files='C:/Users/hyeyoung/PycharmProjects/test/kcbert/all.txt',features=emotion_features) #split='train+test',
dataset['train'][10]
#,column_names=['text_script','multimodal_emotion']
#dataset=load_dataset('all.txt',data_files='C:\Users\hyeyoung\PycharmProjects\test\kcbert\',delimiter='',column_names=['text','label'])
#dataset=load_dataset('text','label',data_files={'train':'C:/Users/hyeyoung/PycharmProjects/test/kcbert/ratings_train.txt','test':'C:/Users/hyeyoung/PycharmProjects/test/kcbert/ratings_test.txt'},split='train+test')
#dataset=load_dataset('text',data_files=['all.txt'],delimiter=';',column_names=['text_script','label'],features=emotion_features)
#dataset=load_dataset('text',data_files=['ratings_train.txt','ratings_test.txt'])
#dataset=load_dataset('text','label',data_files={'train':'ratings_train.txt','test':'ratings_test.txt'})

#dataset
#tokenized_datasets = dataset.remove_columns(["text"])
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')

def tokenize_function(examples): #examples=
    return tokenizer(examples["text"], padding="max_length", truncation=True)
type(dataset) #DatasetDict

#examples=dataset
#tokenized_datasets=list(dataset).map(tokenize_function,batched=True)
#tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#drop

tokenized_datasets.set_format("torch")
from torch.utils.data import DataLoader

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
type(small_train_dataset) #datasets.arrow_dataset.Dataset
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

#train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
#eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=4)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=4)
#-------------여기까지가 dataset 준비-----------------------

#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=8)
model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base',num_labels=8)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        #batch = {k: v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    #batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    print(metric.compute())
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()



# import gc
# gc.collect()
# torch.cuda.empty_cache()






