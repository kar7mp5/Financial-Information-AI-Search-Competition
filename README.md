# 2024-INHA-AI-CHALLENGE
[**2024 인하 인공지능 챌린지**](https://dacon.io/competitions/official/236291/overview/description)


## Table of Contents

### [**1. Structure of the file**](#1-structure-of-the-file)  

### [**2. Install libraries**](#2-install-libraries)  

### [**3. About the train data**](#3-about-the-train-data)  

### [**4. LLM model**](#4-llm-model)
[**4-1. About the model**](#4-1-about-the-model)  
[**4-2. Install the model**](#4-2-install-the-model-not-used)  
[**4-3. Way to use the trained model**](#4-3-way-to-use-the-trained-model)

### [**5. Train history**](#5-train-history)
[**2024.07.03.**](#20240703)
[**2024.07.04.**](#20240704)

---

## **1. Structure of the file**

```
├── content/
│   ├── sample_submisstion.csv  # sample submissition file.
│   ├── test.csv                # The LLM model need to hit the answer from this file.
│   └── train.csv               # The data of the llm model.
│
├── results/                    # Fine-tuned model's history.
│
├── models/                     # Fine-tuned model's in here.
│
├── baseline.ipynb              # Reserved base code from the DACON contest.
├── finetuning.ipynb            # Main code that use in the contest.
└── langchain_test.ipynb        # Langchain test code.
```


## **2. Install libraries**
```console
$ pip3 install -r requirements.txt
```


## **3. About the train data**

PATH: `/content/train.csv`  
FORMAT:
```
id,context,question,answer # HEADER
```


## **4. LLM model**

### **4-1. About the model**
- [~~llama3 8B~~](https://ollama.com/library/llama3:8b)  (Not Used!)  
- [~~EEVE-Korean-Instruct-10.8B-v1.0-GGUF~~](https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF/tree/main) (Not Used!)
- **beomi/Llama-3-Open-Ko-8B** --> **fine-tuning**


### **4-3. Way to use the trained model**
Change the `model_id` to your fine-tuned model.
```python
# load the model.
model_id = "./models/20240703" # <-- CHANGE HERE TO YOUR MODEL PATH
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            torch_dtype="auto", load_in_4bit=True)
```

## **5. Train history**
### 2024.07.03.
**PROMPT**
```
question_prompt = f"너는 주어진 Context를 토대로 Question에 답하는 챗봇이야. \
                    Question에 대한 답변만 한 단어로만 최대한 간결하게 답변 해. \
                    Context: {context} Question: {question}\n Answer:"
```

**RESULT**
```
Answer for question: 산청에서 2019년부터 작물생산교육과 함께 실시해 온 프로젝트는 뭐야 :  산청군은 산림청과 함께 산림소득을 
Processed count: 1
Answer for question: 세계 주요 투자은행은 한국경제가 2021년에 몇 프로 성장할 것이라 예상했어 :  세계 주요 투자은행(IB)은 한국경
Processed count: 2
Answer for question: 개인이 순매수한 삼성전자 주식 금액 :  12조 691억 원
 
Processed count: 3
Answer for question: 닛산이 새롭게 등록한 상표권의 이름은 :  I-파워 
Processed count: 4
Answer for question: 얼마큼의 배당금을 동국제강 주주가 받을 수 있는 거야 :  동국제강은 지난해 결산배당으로 보통
Processed count: 5
Answer for question: 26일 한국갤럽이 한국 경기 전망을 조사한 대상 인원은 몇이나 돼 :  1,001명
```

### 2024.07.04.
**PROMPT**
```
question_prompt = f"너는 주어진 Context를 토대로 Question에 답하는 챗봇이야. \
                    Question에 대한 답변만 한 단어로만 최대한 간결하게 답변 해. \
                    Context: {context} Question: {question}\n Answer:"
```
**RESULT**
```
Processed count: 1
Answer for question: 10일 오전 거래소를 기준으로 비트코인은 얼마에 거래됐어 :  7540만원
 
Processed count: 2
Answer for question: 위플이앤디가 12주 동안 HMR 제품을 협찬하기로 한 곳은 어디야 :  국방 FM '레이나의 건빵과 별사탕
Processed count: 3
Answer for question: 어느 기관에서 부동산 투기로 조직 개편에 들어섰어 :  한국토지주택공사(LH)이번에 새로 
Processed count: 4
Answer for question: 올해 1분기 전국 입주 예정 아파트는 몇 가구인지 :  8만387가구야. 
Processed count: 5
Answer for question: 해양수산부 장관이 나무 수종을 실시한 곳은 어디야 :  정부세종청사 5동 녹지공간
 
Processed count: 6
Answer for question: 이선홍 전주상공회의소 회장이 직무를 수행하는 동안 제일 만족스러웠던 일은 무엇이라고 했어 :  이선홍 회장은 전주상의 회관을 새로
Processed count: 7
Answer for question: GS25가 출시한 빵 브랜드 명칭은 무엇인가 :  브레디크
 
Processed count: 8
Answer for question: LG유플러스가 ESG위원회를 새로 설치하고 위원장 자리를 맡긴 사람이 누구야 :  제현주 사외이사야. 제현주 사외이사
Processed count: 9
Answer for question: 한국농어촌공사에서 9일 진행된 일은 뭐지 :  한국농어촌공사에서 9일 진행된 일은
```

**F1 Score**
```
{'f1': 68.68481305565705}
```

**Real Score**  
0.56343  