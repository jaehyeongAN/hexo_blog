---
title: Keras functional api - Multi-input 모델 구축하기
date: 2019-03-26 00:40:27
tags:
- machinelearning
- deeplearning
- keras
- tensorflow 
- functionalapi
- rnn
- lstm
- embedding
---
Intro
---
지난 한달간 회사 프로젝트를 위해 공부한 내용을 정리할 겸 오늘은 keras functional api(함수형 api)에 대한 소개와 이것을 어떻게 적용하는지를 LSTM모델과 embedding모델을 통해 간단히 소개하려고 한다.

---


그동안 keras를 통해 딥러닝 모델을 구축하기 위해서는 Sequential 모델을 이용하였을 것이다.
Sequential 모델은 네트워크의 입력과 출력이 하나라고 가정하고 층을 차레대로 쌓아 구성한다. 

<img src="/image/sequential.PNG" width="300" height="300">

따라서 위와 같은 Sequential 모델에 데이터를 학습하기 위해서는 모든 데이터를 같은 방식으로 전처리하여 모델에 맞게 shape을 구성해주어야 한다.
하지만, 위와 같은 구성이 맞지 않는 경우도 존재한다. 예를 들어, 중고 의류 시장 가격을 예측하는 딥러닝 모델을 만든다고 가정해보겠다.

이 모델은 시장 가격 예측을 위해 의류 브랜드, 제작 연도와 같은 정보(메타 데이터), 사용자가 제공한 제품 리뷰(텍스트 데이터), 해당 의류 사진(이미지 데이터)과 같은 데이터를 받는다.
<img src="/image/cloth_example.PNG" width="400" height="400">

모델은 데이터의 특성에 맞게 적절히 사용되어야 하는데, 해당 데이터가 text인지, image인지, time-series인지에 따라 학습하는 모델도 달라진다. 
위와 같은 경우, 
<img src="/image/cloth_model.PNG" width="400" height="400">
메타 데이터만 있다면 이를 one-hot encoding하여 단순한 DenseNet모델을 구현할 수 있을 것이고, 
텍스트 데이터의 경우 이를 word2vec 같은 기법을 통해 벡터로 변환하여 Embedding 모델이나 혹은 RNN모델을 구현할 수 있을 것이고,
이미지 데이터의 경우 CNN과 같은 ConveNet 모듈을 이용하여 데이터를 학습할 수 있을 것이다.

keras functional api
---
하지만 방금 살펴본 것과 같이 예측에 사용되는 데이터가 여러 형태로 존재한다면 어떤 모델을 사용해야 할까? 
단순히 텍스트와 이미지를 vectorize하여 예측 변수로 추가하여 사용해야 할까? 
데이터 특성에 따라 각각 모델을 학습시킬 순 없을까?

이러한 의문을 해결해줄 것이 바로 오늘 살펴볼 **Keras Functional API**이다 
함수형 api라고 불리며, 말 그대로 모델을 함수처럼 필요할 때 호출하여 사용할 수 있도록 한다. 즉, 모델을 함수로 구현하여 모듈식으로 이용한다는 말이다.

다시 위의 예로 돌아가 함수형 API를 활용하면 아래 그림과 같이 모델별 학습 및 예측이 가능해진다.
<img src="/image/cloth_model_concat.PNG" width="500" height="500">


위 그림과 같은 모델을 다중입력모델(multi-input model)이라고 하며 이 외에도 다중출력모델(multi-output model)이 존재합니다. 
- 다중입력모델: 데이터 특성에 따른 서로 다른 여러개의 모델이 input으로 사용되어 하나의 output을 내는 네트워크
<img src="/image/multi_input.PNG" width="400" height="400">
- 다중출력모델: 하나의 output이 아닌 데이터에 있는 여러 속성을 동시에 예측하는 네트워크
<img src="/image/multi_output.PNG" width="400" height="400">

---
함수형 API는 기존 구현방법과 구조적으로 차이가 있다.
보통 모델을 구현할 때 Sequential()객체를 생성 후 시퀀스 형태로 순차적으로 layer를 쌓아가지만 함수형 api는 Model()객체를 통해 모델을 구현한다. 

- 기존 Sequential() 사용 시 
```python
from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.fit(data, labels)  # starts training
```

- funciontal api() 사용 시
```python
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=outputs)
```
위와 같이 Sequential()객체는 input부터 output까지 순차적으로 이루어지지만, 함수형 api는 각각의 변수에 layer를 받아 모듈별로 구성할 수 있으며, 마지막에는 Model()객체에 input과 output텐서를 지정하여 모델을 생성한다.


적용
---
그렇다면 직접 keras를 이용하여 적용하는 과정을 소개하려고 한다. 
데이터 셋과 전처리 과정은 공개할 순 없으나 해당 데이터는 일반 Sequence 데이터 및 Text 데이터로 이루어져 있고, 고장발생에 대한 여부를 예측하는 문제이다. 

functional api를 적용하기 위하여 두개의 모델을 구축하였다.
- Sequence 데이터를 위해서는 시간 및 순서가 있는 데이터에 효율적인 LSTM(Long Short Term Memory Network)를 이용하였고, 
- text 데이터는 vectorize 후 Embedding 모델을 이용하였다.

개략적인 모델 구성도는 대략 아래 그림과 같다. 
<img src="/image/model_structure_0.PNG" width="400" height="400">
<p></p>
**1. LSTM 모델 적용을 위한 Sequence 데이터 처리**
우선 LSTM과 같은 Recurrent 모델은 크기가 (timesteps, input_features)인 2D 텐서로 인코딩된 벡터의 시퀀스를 입력받기 때문에 shape을 맞추어 준다. 
shape을 맞춰주기 전에 우선 text변수와 target 값을 제외해준 후, 데이터를 normalize해주었다.

```python
## preprocessing for lstm 
sequence_train = df[:train_size].drop(['text_data','target'], axis=1)
sequence_test = df[train_size:].drop(['text_data','target'], axis=1)

# normalize
scaler = StandardScaler().fit(sequence_train)
sequence_train_scale = scaler.transform(sequence_train)
sequence_test_scale = scaler.transform(sequence_test)

timesteps = 1
columns_size = len(sequence_train.columns)
sequence_train = sequence_train_scale.reshape((sequence_train_scale.shape[0], timesteps, columns_size))
sequence_test = sequence_test_scale.reshape((sequence_test_scale.shape[0], timesteps, columns_size))
```
<p></p>
**2. Embedding 모델 적용을 위한 text 데이터 처리**
Embedding 모델을 구현하기 위하여 먼저 데이터를 3D 텐서로 변환시켜주어야 한다. 이를 위해 keras의 Tokenizer()객체를 이용하였다. 과정은 아래와 같다.
- fit_on_texts(): 텍스트 데이터를 통해 word index를 구축
- texts_to_sequences(): word index를 통해 해당 텍스트를 시퀀스 형태로 변환
- pad_sequences(): 3D 텐서로 변환하기 위해 padding을 추가

```python
## preprocessing for embedding
text_embed = df.loc[:, ['text_data']]
text_embed_train = text_embed[:train_size]
text_embed_test = text_embed[train_size:]

# tokenize
max_words = 1000	# 사용할 최대 단어 수 
max_len = 50		# 단어의 길이
tokenizer = text.Tokenizer(num_words=max_words) 	# top 1,000 words
tokenizer.fit_on_texts(text_embed_train)			# word_index 구축
sequences_text_train = tokenizer.texts_to_sequences(text_embed_train)	# return sequence
sequences_text_test = tokenizer.texts_to_sequences(text_embed_test)		
# add padding 
pad_train = sequence.pad_sequences(sequences_text_train, maxlen=max_len)# return 3D tensor
pad_test = sequence.pad_sequences(sequences_text_test, maxlen=max_len)	# return 3D tensor
```
<p></p>
**3. multi-input model 구축**
우선 LSTM모델과 Embedding모델을 만든 후 concatenate(model1, model2)함수를 이용하여 서로 다른 두 개의 모델의 output을 하나의 모델로 통합할 수 있다. 

```python
def multi_input_lstm_embedding_model(timesteps, columns_size, max_words, max_len):
	# lstm model
	lstm_input = layers.Input(shape=(timesteps, columns_size))
	lstm_out = layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)(lstm_input)

	lstm_model = Model(inputs=lstm_input, outputs=lstm_out)

	# embedding model 
	embed_input = layers.Input(shape=(None,))
	embed_out = layers.Embedding(max_words, 8, input_length=max_len)(embed_input)
	embed_out = layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3))(embed_out)

	embed_model = Model(inputs=embed_input, outputs=embed_out)

	# concatenate
	concatenated = layers.concatenate([lstm_model.output, embed_model.output])
	concatenated = layers.Dense(32, activation='relu')(concatenated)
	concatenated = layers.BatchNormalization()(concatenated)
	concat_out = layers.Dense(2, activation='sigmoid')(concatenated)

	concat_model = models.Model([lstm_input, embed_input], concat_out)

	return concat_model

## model define
concat_model = multi_input_lstm_embedding_model(timesteps, columns_size, max_words, max_len)
concat_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model fit
concat_model.fit([df_label_train, pad_train], target_train
				epochs=7, batch_size=32,
				callbacks=callbacks_list,
				validation_data=([sequence_test, pad_test], target_test),
				shuffle=False)	# because of time-series
```

---
Outro
---
keras functional api를 이용한다면 좀 더 데이터 특성에 유연하게 모델을 학습시킬 수 있다는 것이 큰 장점인 것 같다. 