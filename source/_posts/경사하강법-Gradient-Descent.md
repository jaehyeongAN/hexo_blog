---
title: 경사하강법(Gradient Descent)
date: 2019-04-23 21:52:20
tags:
- learningrate
- gradientdescent
- batchgd
- minibatchgd
- sgd
- machinelearning
- deeplearning
- meansquarederror
---
Intro
--- 
최적의 예측 모델을 만들기 위해서는 실제값(true)과 예측값(predict)과의 Error(cost function)가 최소가 되는 모델을 찾는 것이다. 하지만 분석자가 직접 모델의 Cost function을 최소화시키는 파라미터 값을 찾기 위해서는 수십 번의 파라미터 변경이 필요하기 때문에 모델이 학습과정에서 스스로 cost function이 최소가 되도록 파라미터를 조정해나가는 경사하강법(Gradient Decent)이 사용된다.  

---

Gradient Descent
---
#### 경사하강법(Gradient Descent)
<img src="/image/gradient_descent.png" width="500" height="400">

경사하강법이란 비용함수(Cost Function)을 최소화하기 위하여 반복적해서 파라미터를 조정해나가는 것을 말한다. 
만약 한 밤 중에 산에서 길을 잃었을 때, 산 밑으로 내려가는 가장 좋은 방법은 무엇일까? 바로 가장 가파른 길을 따라 산 아래로 내려가는 것이다. 이와 같이 최적의 값에 도달하기 위해 가장 빠른 길을 찾는 과정을 경사 하강법의 기본원리라고 할 수 있다.
-- 파라미터 벡터 theta()에 대해 cost function의 현재 gradient를 계산
-- theta()의 경우 임의의 값으로 시작해서(random initialization) 조금씩 cost function이 감소되는 방향으로 진행 


#### 학습률(learning rate)
경사 하강법에서 중요한 파라미터로서 학습 시 스템(step)의 크기


* 학습률이 너무 작을 경우
 -- 알고리즘이 수렴하기 위해 반복을 많이 진행해야 하므로 학습 시간이 오래걸림
 -- 지역 최솟값(local minimum)에 수렴할 수 있음 

* 학습률이 너무 클 경우 
 -- 학습 시간이 적게 걸리나
 -- 스텝이 너무 커 전역 최솟값(global minimum)을 가로질러 반대편으로 건너뛰어 최솟값에서 멀어질 수 있음

<img src="/image/learning_rate_sl.png" width="500" height="400">


#### 경사하강법의 문제점
-- 무작위 초기화(random initialization)으로 인해 알고리즘이 전역 최솟값이 아닌 지역 최솟값에 수렴할 수 있음
-- 평탄한 지역을 지나기 위해선 시간이 오래 걸리고 일찍 멈추게 되어 전역 최솟값에 도달하지 못할 수 있음
-- 하지만 선형 회귀(Linear Regression)를 위한 MSE(Mean Squared Error) cost function은 어떤 두점을 선택해 어디에서 선을 그어도 곡선을 가로지르지 않는 볼록 함수(convex function)임
-- 이는 지역 최솟값이 없고 하나의 전역 최솟값만을 가지는 것을 뜻하며, 연속된 함수이고 기울기가 갑자기 변하지 않음 
<img src="/image/convex_nonconvex.jpg" width="500" height="400">
<p></p>
<p></p>

Batch Gradient Descent
---
-- 경사하강법을 구현하려면 각 모델 파라미터 theta()에 대해 비용 함수의 그래디언트를 계산해야 함.
-- 즉, theta()가 조금 변경될 때 비용함수가 얼마나 변하는지 계산해야 하는데 이를 편도 함수(partial derivative)라고 함.
<img src="/image/partial_derivative.JPG" width="400" height="300">
-- 매 gradient descent step에서 훈련 데이터 전체를 사용
-- 그렇기 때문에 매우 큰 training set에서는 학습이 매우 느림 
<p></p>
<p></p>

Stochastic Gradient Descent(SGD)
---
<img src="/image/sgd.png" width="550" height="500">

-- 매 step에서 딱 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플에 대한 gradient를 계산
-- 매우 적은 데이터를 처리 하기 때문에 학습 속도가 빠르고, 하나의 샘풀만 메모리에 있으면 되므로 매우 큰 training set도 훈련이 가능
-- cost function이 매우 불규칙할 경우 알고리즘이 local minimum을 건너뛰도록 도와주므로 global minimum을 찾을 가능성이 높음
-- 하지만 샘플 선택이 확률적(Stochastic)이기 때문에 배치 경사 하강법에 비해 불안정 
-- cost function이 local minimum에 다다를 때까지 부드럽게 감소하지 않고  위아래로 요동치면서 평균적으로 감소
<p></p>
<p></p>

Mini-batch Gradient Descent
---
-- mini-batch라 불리는 임의의 작은 샘플 세트에 대해 gradient를 계산
-- SGD에 비해 matrix 연산에 최적화되어 있으며, 파라미터 공간에서 덜 불규칙하게 학습
-- 하지만, local minimum에 빠지면 빠져나오기 힘듬


--- 
Outro
---
아래는 batch, mini-bath, stochastic gradient descent의 경사 하강법 진로를 살펴본 그림이다.
<img src="/image/sgd_mini_batch.png" width="550" height="500">
(출처 : 핸즈온 머신러닝)

모두 최솟값 근처에 도달했지만 배치 경사 하강법의 경로가 실제로 최솟값에서 멈춘 반면 확률적 경사 하강법 및 미니배치 경사하강법은 근처에서 맴돌고 있다. 그렇지만 배치 경사 하강법에는 매 스텝에서 많은 시간이 소요되고, 확률적 경사 하강법과 미니배치 경사 하강법도 적절한 학습 스케쥴(learning schedule)을 사용하면 최솟값에 도달할 수 있다.
