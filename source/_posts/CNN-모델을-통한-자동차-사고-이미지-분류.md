---
title: CNN 모델을 통한 자동차 사고 이미지 분류
date: 2018-07-01 12:50:26
tags:
- cnn
- deeplearning
- keras
- crawling
---
Intro
---
회사 프로젝트에서 **자동차 사고 이미지 분류 모델**을 만들 일이 생겨 CNN 모델을 적용한 과정을 정리해 보고자 합니다. 
전체적으로 크롤링(crawling)을 통해 사고 이미지를 수집하였으며, 수집한 데이터를 바탕으 아래 7개의 사고를 분류하는 **다중(multi class) 분류 모델**을 생성하였습니다.
1. 전방 추돌(Car front crash)
2. 측면 추돌(Car side crash)
3. 후방 추돌(Rear and crash)
4. 유리창 깨짐(Car broken windshield)
5. 차 스크래치Car scratch)
6. 타이어 펑크(Flat tire)
7. 전복 (Overturned vehicle)

사고 이미지 데이터 수집
---
그 동안 크롤링을 할때 python의 **lxml의 parse 함수**를 이용하여 html 태그 기반으로 데이터를 수집하였는데, 정말 간딴하게! 구글에서 이미지를 수집할 수 있는 라이브러리인 **[icrawler](https://icrawler.readthedocs.io/en/latest/builtin.html)**를 알게 되어 쉽게 이미지를 수집할 수 있었습니다.

아래와 같이 icrawler의 GoogleImageCrawler()를 이용하였습니다.
```python
from icrawler.builtin import GoogleImageCrawler
google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': '../data'})

google_crawler.crawl(keyword='car crash', max_num=500,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)

```
- keyward: 수집하고자 하는 이미지
- max_num: 수집할 이미지 수
- date_min/date_max: 수집할 기간
- min_size/max_size: 이미지 크기 

이후, 수집한 데이터를 이미지 처리 및 train/test set으로 나누었습니다.
```python
rom PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 선택하기 
accident_dir = "./image"
categories = ["Car front crash","Car side crash","Rear and crash","Car broken windshield","Car scratch","Flat tire","Overturned vehicle"]
nb_classes = len(categories)
# 이미지 크기 지정 
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = accident_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

print('>>> data 저장중 ...')
np.save("./image/7obj.npy", xy)
print("ok,", len(Y))

```
이미지를 RGB로 변환 후,  64x64 크기로 resize해주었습니다.


CNN 모델 생성
---
모델은 이미지 분류의 정석으로 불리는 **CNN(Convolution Neural Network)** 모델을 활용하였습니다.
총 3개의 층으로 구성하였고, 활성화 함수로는 relu 및 softmax 함수를 적용하였습니다. dropout도 적용하여 과적합을 방지하였습니다.
![car_cnn](/image/car_cnn.png)

모델 학습 후 **.hdf5** 파일로 저장합니다.
```python
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

# 카테고리 지정하기
categories = ["Car front crash","Car side crash","Rear and crash","Car broken windshield","Car scratch","Flat tire","Overturned vehicle"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("./image/7obj.npy")
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 학습 완료된 모델 저장
hdf5_file = "./image/7obj-model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, batch_size=32, nb_epoch=10)
    model.save_weights(hdf5_file)
```
모델의 오차와 정확도를 살펴보겠습니다.
```python
# 모델 평가하기 
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc
```
![image](/image/cnn_score.jpg)
**오차는 0.03, 정확도는 98%** 정도의 성능을 나타냅니다. 확실히 데이터를 많이 학습시키니 성능이 좋은 것 같습니다.

신규 데이터 예측
---
**학습된 모델(7obj-model.hdf5)**에 신규 이미지를 적용하여 이미지의 클래스를 예측해보도록 하겠습니다. 

적용할 이미지는 아래의 **차 전복(Overturned vehicle)** 이미지 입니다. 
![overturned](/image/overturned.jpg)

모델에 적용해봅니다.
```python
# 적용해볼 이미지 
test_image = './image/test_overturned.jpg'
# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((64,64))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 64, 64,3)
# 예측
pred = model.predict(X)  
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
print('New data category : ',categories[result[0]])
```
학습할때와 똑같이 이미지를 처리해 주고 저장된 모델을 통해 이미지를 예측합니다. 

- 예측결과
```python
New data category : Overturned vehicle
```

Overturned Vehicle(차 전복) 클래스로 이미지가 모델에 의해 예측되었습니다! 모델이 잘 학습된 것 같습니다. 
각 클래스별로 500개 총 3500개의 이미지를 통해 학습한 CNN 다중 분류 모델의 성능이 생각보다 괜찮은 것 같습니다. 