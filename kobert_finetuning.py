import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW # 인공지능 모델의 초기값 지정 함수를 아담으로 지정한다.
from transformers.optimization import get_cosine_schedule_with_warmup

#GPU 사용
device = torch.device("cuda:0")

#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

import pandas as pd
data=pd.read_csv('/content/drive/My Drive/kobert/mergeAll.tsv',delimiter='\t') #바꿔야함.

data.sample(n=10)




