---
title: 선형 모델(Linear Model)
date: 2019-04-25 15:27:13
tags:
- regression
- linear
- ridge
- lasso
- elsasticnet
---
Intro
--- 
머신러닝을 원리를 이해하기 위해 가장 먼저 배우게 되는 선형 모델(linear models)에 대한 글이다. 선형 모델은 100여 년 전게 개발되었고, 지난 몇십 년 동안 폭넓게 연구되고 현재도 널리 쓰이고 있다. 기본적으로 선형 모델은 입력 특성에 대한 선형 함수를 만들어 예측을 수행한다.

---
Linear Regression(선형 회귀)
---
회귀의 경우 선형 모델을 위한 일반화된 예측 함수는 아래와 같다. 

<img src="/image/linear_.JPG" width="500" height="400">

위 식에서 x[0]부터 x[n]까지는 하나의 데이터 포인트에 대한 특성을 나타내며(특성의 개수는 n+1), w와 b는 모델이 학습할 파라미터이다.그리고 y^은 모델이 만들어낸 예측값이다.
위 식은 특성이 하나인 데이터 셋이라면 아래와 같이 1차 방정식으로 단순하게 나타낼 수 있다. 

<img src="/image/linear_2.JPG" width="130" height="400">

w[0]는 기울기이고, b는 y축과 만나는 절편(또는 편향)이다. 특성이 많아지면 w는 각 특성에 해당하는 기울기를 모두 가진다. 

<img src="/image/linear_regression.png" width="400" height="400">
[선형 회귀]

<p></p>
선형 회귀는 가장 간단하고 오래된 회귀용 선형 알고리즘이다. 선형 회귀는 예측 값 y^과 실제 값 y 사이의 평균제곱오차(mean squared error)를 최소화 하는 파라미터 w와 b를 찾는다. 평균제곱 오차는 예측값(y^)과 실제값(y)의 차이를 제곱하여 더한 후에 샘플 개수로 나눈 값이다.

<img src="/image/mse.JPG" width="350" height="400">

<p></p>
아래는 scikit-learn의 LinearRegression을 통해 boston house price를 통한 집 값 예측을 수행하는 코드이다.

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# dataset
X, y = load_boston(True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

####### Linear Regression #######
lin_reg = LinearRegression().fit(X_train, y_train)
train_pred = lin_reg.predict(X_train)
test_pred = lin_reg.predict(X_test)

print('MSE of train set: ', mean_squared_error(y_train, train_pred)) # 19.640
print('MSE of test set: ', mean_squared_error(y_test, test_pred))    # 29.782
```
training set과 test set에 대한 MSE가 각각 19.6, 29.7로 나타났다. 이는 모델이 training set에 과대적확(Overfitting)되었다는 이야기다. 하지만 선형회귀에서는 이런 과대적합을 방지할 규제(regularization)방안이 없다.
때문에 규제 방안이 포함되어 있는 알고리즘(Ridge, Lasso, ElasticNet 등)을 이용하는 것이 효율적일 수 있다. 이는 좀 더 아래에서 살펴볼 것이다.  
<br /> 

Polynomial Regression(다항 회귀)
---
가지고 있는 데이터가 단순한 직선보다 복잡한 형태라면 어떻게 선형회귀를 적용해야 할까? 신기하게도 비선형(Non-lieaner) 데이터를 학습하는 데 선형 모델을 사용할 수 있다. 

바로 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 이 확장된 특성을 포함한 데이터셋에 선형 모델을 훈련시키는 것이다. 이런 기법을 다항 회귀(Polynomial Regression)이라고 한다.

만약 아래와 같이 임의로 만든 2차 방정식의 비선형 데이터가 있다고 해보자.
```python
m = 100
X = 6 * np.random.rand(m,1)-3
y = 0.5*X**2+X+2+np.random.randn(m,1)
```
<img src="/image/polynomial_data.png" width="400" height="400">

위와 같은 데이터가 non-linear 데이터인데 선형회귀 모델을 위 데이터에 적용해보면, 
```python
lin_reg = LinearRegression()
lin_reg.fit(X, y)
pred = lin_reg.predict(X)
```
<img src="/image/predict_poly.JPG" width="400" height="400">

위 그림과 같이 데이터의 비선형적 패턴을 전혀 파악하지 못한 채 1차 직선으로만 예측을 하게 된다.
이제 이러한 비선형 데이터를 선형 예측하기 위해 위 데이터에 각 특성을 제곱하여 새로운 특성을 추가한다. 

```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
pred = lin_reg.predict(X_poly)
```
<img src="/image/poly_predict.png" width="400" height="400">
데이터에 새로운 다항 특성을 추가하였을 때 선형 모델이 데이터의 패턴을 파악하여 예측하는 특성을 보여주고 있다. 
위 데이터의 실제 함수는,
<img src="/image/p_li_1.JPG" width="230" height="400">
이고, 예측 모델의 함수는,
<img src="/image/p_li_2.JPG" width="250" height="400">
이므로 실제 값과 예측 값의 차이가 많지 않음을 알 수 있다.
<br /> 

Regularized Linear Regression(규제가 있는 선형 모델)
---
위에서 살펴보았듯 이 선형 모델의 경우 모델이 훈련 데이터 셋에 과대적합(overfitting)되더라도 모델을 규제할 방안이 없다.
따라서, 과대적합을 감소시키기 위해서는 모델의 가중치(weight)를 규제(제한)함으로써 과대적합되기 어렵게 만들어야 한다.

이렇게 가중치를 제한하는 알고리즘으로 릿지(Ridge), 라쏘(Lasso), 엘라스팃넷(ElasticNet) 회귀에 대해 살펴보려고 한다. 


#### 1. Ridge Regression(릿지 회귀)
선형 회귀에 규제가 추가된 회귀 모델이다. 규제항이 비용함수에 추가 되며 이는 학습 알고리즘을 데이터에 맞추는 것뿐만 아니라 모델의 가중치가 가능한 한 작게 유지되도록 한다. 규제항은 훈련하는 동안에만 비용함수에 추가되고, 훈련이 끝나면 모델의 성능을 규제가 없는 성능 지표로 평가한다.

 -- 선형회귀에 규제(L2: 가중치들의 제곱합을 최소화)를 걸어 과대적합을 방지
 -- 하이퍼파라미터 a(alpha)는 모델을 얼마나 규제할지 조절 
 -- a = 0 이면, 릿지 회쉬는 선형회귀와 같음
 -- a 가 아주 크면 모든 가중치가 거의 0에 가까워지고 결국 데이터의 평균을 지나는 수평선이 됨
 <br /> 
 릿지 회귀의 비용함수는 아래 수식과 같다.
 <img src="/image/ridge_mse.JPG" width="350" height="400">

 위에서 보았던 boston house price 예측에 ridge 회귀를 적용할 경우 선형 회귀보다 더 나은 성능을 얻을 수 있다.

```python
####### Ridge Regression ####### 
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1).fit(X_train, y_train)
train_pred = ridge.predict(X_train)
test_pred = ridge.predict(X_test)

print('MSE of train set: ', mean_squared_error(y_train, train_pred)) # 19.645
print('MSE of test set: ', mean_squared_error(y_test, test_pred))    # 29.878
```
<br /> 

#### 2. Lasso Regression(라쏘 회귀)
라쏘 회귀 역시 선형 회귀에 규제가 추가된 모델이며, 릿지 회귀에서 사용된 L2 규제가 아닌 가중치 벡터의 L1 노름을 사용한다. 라쏘 회귀의 가장 중효한 특징은 덜 중요한 특성의 가중치를 완전히 제거하려고 한다는 것이다. 

 -- 자동으로 덜 중요한 특성을 제거하는 특성 선택(feature selection)을 수행하고 희소 모델(spare model)을 만듬(즉, 0이아닌 특성의 가중치가 작음)
 -- 이를 통해 모델을 이해하기 쉬워지고 모델의 가장 중요한 특성이 무엇인지 파악 가능
 <p></p>
 라쏘 회귀의 비용함수는 아래 수식과 같다.
 <img src="/image/lasso_mse.JPG" width="350" height="400">

 마찬가지로 boston house에 Lasso 모델을 적용한다.  lasso모델의 coef_ 파라미터를 이용하면 몇 개의 특성이 제외되고 사용되었는지 알 수 있다.

```python
####### Lasso Regression #######
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
train_pred = lasso.predict(X_train)
test_pred = lasso.predict(X_test)

print('MSE of train set: ', mean_squared_error(y_train, train_pred)) # 19.678
print('MSE of test set: ', mean_squared_error(y_test, test_pred))    # 30.091
print('사용한 특성의 수 : ',np.sum(lasso.coef_ != 0))					 # 13
```
<br /> 

#### 3. Elastic Net(엘라스틱넷)
엘라스틱넷은 릿지 회귀와 라쏘 회귀를 절충한 모델이다. 규제항은 릿지와 회귀의 규제항을 단순히 더해서 사용하며, 혼합 정도는 혼합 비율 r을 사용해 조절한다. 
 -- r = 0이면, 엘라스틱넷은 릿지 회귀와 같고,
 -- r = 1이면, 라쏘 회귀와 같아짐

 <p></p>
 엘라스틱넷의 비용함수는 아래 수식과 같다.
 <img src="/image/elasticnet_mse.JPG" width="350" height="400">

 elastic net도 boston house price에 적용.
```python
####### ElasticNet #######
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.001, max_iter=10000000).fit(X_train, y_train)
train_pred = elastic.predict(X_train)
test_pred = elastic.predict(X_test)

print('MSE of train set: ', mean_squared_error(y_train, train_pred)) # 19.657
print('MSE of test set: ', mean_squared_error(y_test, test_pred))    # 29.974
```


Outro
--- 
그렇다면 선형회귀, 릿지, 라쏘, 엘라스틱넷을 각각 언제, 어떤 상황에 사용해야 좋을까?
적어도 규제가 있는 모델이 대부분의 상황에서 좋으므로 일반적으로 선형회귀 모델을 사용하는 것은 피하는 것이 좋다.

 -- 기본적으로 ridge 회귀가 기본이 되어 사용 됨
 -- 하지만, 특성이 많고 그 중 일부분만 중요하다면 lasso나 elastic net이 더 좋은 선택일 수 있음
 -- 또한, 특성 수가 훈련 샘플 수보다 많거나 특성 몇 개가 강하게 연관되어 있을 때는 lasso보다는 elastic net이 선호 됨