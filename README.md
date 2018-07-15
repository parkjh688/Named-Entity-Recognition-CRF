## Named Entity Recognition using CRF in Python
---
Conditional random fields (CRFs) 라이브러리인 [python-crfsuite](http://python-crfsuite.readthedocs.io/en/latest/index.html) 를 이용해서 named entity recognition 을 하는 코드입니다.


> CRFs 란 통계적 모델링 방법중에 하나로, 패턴 인식과 기계 학습과 같은 구조적 예측에 사용된다. 일반적인 분류자(영어: classifier)가 이웃하는 표본을 고려하지 않고 단일 표본의 라벨을 예측하는 반면, 조건부 무작위장은 고려하여 예측한다. 자연 언어 처리 분야에서 자주 사용되는 선형 사슬 조건부 무작위장(영어: linear chain CRF)은 일련의 입력된 표본들에 대해 일련의 라벨들을 예측한다.
<출처 : 위키피디아>


POS-tagging 에는 konlpy 의 Twitter 를 사용하였습니다.
train, predict 를 하는 함수의 사용 방법 및 input, output 은 crf-example.ipynb 파일을 참고하세요.

<br>
#### Installation
---
`$ pip install -r requirements.txt`

<br>
#### Result
---
아래는 pandas 로 표현한 raw data 입니다.
![data](https://i.imgur.com/H7mToDE.png)

<br>

train 후 crf 파일을 load 해서 test data 를 preict 해보면 아래와 같습니다.

- '로스트 치킨 샌드위치 세트 주문할까?' 의 ner 결과 : ['B-SANDWICH', 'I-SANDWICH', 'I-SANDWICH', 'O', 'O', 'O', 'O']
- '스파이시 이탈리안 맵지 않냐' 결과 :['B-SANDWICH', 'I-SANDWICH', 'I-SANDWICH', 'O', 'O', 'O'] 과 같습니다.

![result](https://i.imgur.com/X3ObXad.png)
