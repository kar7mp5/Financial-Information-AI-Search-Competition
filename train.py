import os
import json
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator

# 파일 경로에서 데이터를 읽어옴
file_path = './content/train.csv'
train_data = pd.read_csv(file_path)

# 데이터를 셔플하고 인덱스를 재설정
train_data = train_data.sample(frac=1).reset_index(drop=True)

# 검증 데이터와 학습 데이터로 분할
val_data = train_data[:10]
train_data = train_data[10:]

# 검증 데이터에서 질문과 답변 컬럼을 선택
val_label_df = val_data[['question', 'answer']]

# 학습 데이터를 datasets의 Dataset으로 변환
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_label_df)

# 필요한 라이브러리 불러오기
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch

# torch_dtype 설정
torch_dtype = torch.float16  # 예시로 torch.float16 사용. 필요에 따라 변경 가능

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B",
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
              "beomi/Llama-3-Open-Ko-8B",
              trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import LoraConfig

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # epoch는 1로 설정
    max_steps=7000,  # max_steps을 5000으로 설정
    # 리소스 제약때문에 batch size를 타협해야하는 경우가 발생 -> micro batch size를 줄이고,
    # accumulated step을 늘려, 적절한 size로 gradient를 구해 weight update
    # https://www.youtube.com/watch?v=ptlmj9Y9iwE
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    warmup_steps=150,  # warmup_steps을 절대값으로 설정
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    push_to_hub=False,
    report_to='tensorboard',
)

trainer = SFTTrainer(
    model=model,
    args=training_params,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_params,
    dataset_text_field="question",  # 여기에 적절한 필드 이름을 지정
    # 필요한 추가적인 파라미터들
)

trainer.train()

trainer.save_model('./models/20240704')