---
title: 데이터 분석을 위한 기초 시각화 with Python
date: 2019-08-13 23:00:16
tags:
- visualization
- analytics
- matplotlib
- seaborn
- sklearn
- pca
- correaltion
- featureimportances
- breastcancer
---
Intro
---
데이터를 분석하려는데 데이터의 row와 columns 수가 많은 수백 차원 데이터의 경우 데이터를 파악하기가 쉽지 않다. 그렇기에 인간이 이해할 수 있는 정도의 차원으로 줄여 데이터를 개략적으로 파악하는 것이 필요하고, 역시 인간은 읽고, 듣는 것 보다는 눈으로 보는게 확실히 기억에 오래남고 이해하기 쉽기 때문에 데이터를 시각화하여 분석하는 것이 필요하다.

이번에는 데이터 분석에 앞서 기초적이지만 필수적으로 살펴보아 할 시각화 방법에 대해 살펴볼 것이며, 목록은 아래와 같다.
> 
- *변수 별 데이터 분포(Data Distribution)*
- *타겟 별 2차원 및 3차원 시각화(2D and 3D plot)*
- *변수 간 상관관계(Corrleation)*
- *변수 중요도(Featrue Importances)*

---
예제로 사용 할 데이터로 [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)이다. 많이들 알다시피 유방암에 대해 양성/음성을 예측하기 위한 데이터셋이며, 총 569 row와 31 columns을 가지고 있다. 

## 0. Load Data
우선 데이터를 로드 시킨 후 분석에 불필요한 칼럼은 제외시킨다. 
데이터는 아래와 같은 형태로 되어 있다.
```python
import pandas as pd 

cancer = pd.read_csv('./input/data.csv')
cancer.drop(['id','Unnamed: 32'], axis=1, inplace=True)
cancer.head(10)
```
<img src="/image/cancer_head.JPG" width="1200" height="400">
<br/>

이 데이터에서 예측해야하는 타겟 칼럼은 **'diagnosis'**이며, 'M'은 malignant로 양성을 의미하며, 'B'는 Benign으로 음성을 의미한다.
```python
cancer['diagnosis'].unique() 	# array(['M', 'B'], dtype=object)
```
<br/>

## 1. Column distribution by target
먼저 시각화 해 볼 것은 칼럼 별로 데이터 분포를 시각화해 보는 것이다. 이를 통해 각 칼럼 별로 데이터가 어떻게 분포되어 있는지를 파악할 수 있고, 우리가 예측하고자 하는 타겟(diagnosis)별로 분포가 어떻게 다르게 나타나는지도 파악이 가능하다. 

seaborn의 distplot을 통해 타겟 칼럼인 diagnosis별로 6번째 칼럼까지만 출력해보았다.
```python
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) # 한글 출력 설정 부분

for cnt, col in enumerate(cancer):
    try:
        plt.figure(figsize=(10, 5))
        sns.distplot(cancer[col][cancer['diagnosis']=='M'])
        sns.distplot(cancer[col][cancer['diagnosis']=='B'])
        plt.legend(['malignant','benign'], loc='best')
        plt.title('histogram of features '+str(col))
        plt.show()

        if cnt >= 6: # 6개 칼럼까지만 출력
            break

    except Exception as e:
        pass
```
<img src="/image/cancer_distplot.png" width="900" height="400">

위의 그림으로 보았을 때 radius_mean, area_mean, perimeter_mean 칼럼이 양성일때와 음성일때 분포가 크게 다른 것을 알 수 있고, 특히 area_mean 칼럼은 분포가 넓게 퍼져있는 것을 알 수 있다.
<br/>

## 2. 2 Dimension Plot
이번에는 지난 글에서 살펴보았던 [차원축소(Dimensionality Reduction)](https://jaehyeongan.github.io/2019/05/27/Dimension-Reduction/) 기법을 이용하여 2차원으로 데이터를 시각화하는 방법에 대해 알아보겠다.

우선 데이터 스케일 및 차원축소 기법인 PCA(Principal Component Analysis)를 적용하여 데이터를 2차원으로 변환시켜준 후, 타겟(음성/양성)별로 데이터를 구분하여 출력하였다.

```python
from sklearn.preprocessing import StandardScaler

# Data Scaling
X = cancer.drop(['diagnosis'], axis=1)
y = cancer['diagnosis']

scaler = StandardScaler()
cancer_scale = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```
```python
# plot 2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca2 = PCA(n_components=2)
data_pca2 = pca2.fit_transform(cancer_scale)

plt.figure(figsize=(12, 8))
plt.scatter(data_pca2[:,0], data_pca2[:,1], c=cancer['diagnosis'], s=40, edgecolors='white')
plt.title("2D of Target distribution by diagnosis")
plt.xlabel('pcomp 1')
plt.ylabel('pcomp 2')
plt.show()
```
<img src="/image/cancer_2dplot.png" width="700" height="400">

2차원으로 표현해본 결과 양성일때와 음성일 때 극명하게 분포가 나뉘는 것을 확인해볼 수 있다. 
<br/>

## 3. 3 Dimension Plot
위와 같은 방식으로 PCA를 이용하여 데이터를 3차원으로 변환 후 데이터를 타겟(양성/음성) 별로 시각화 한다.
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3)
data_pca3 = pca3.fit_transform(cancer_scale)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca3[:,0], data_pca3[:,1], data_pca3[:,2], c=cancer['diagnosis'], s=60, edgecolors='white')
ax.set_title('3D of Target distribution by diagnosis')
ax.set_xlabel('pcomp 1')
ax.set_ylabel('pcomp 2')
ax.set_zlabel('pcomp 3')
plt.show()
```
<img src="/image/cancer_3dplot.png" width="700" height="400">
<br/>

## 4. Corrleation Heatmap
이번에는 상관관계 분석을 통해 변수 간 상관관계가 얼마나 있는지 파악해본다. 이러한 상관관계 분석을 통해 타겟 값을 제외한 특정 두 변수가 상관관계가 0.9 이상일 경우 두 변수 중 하나를 제거해주는 것이 좋으며, 또한 어떤 변수가 타겟 값과 높은 상관성을 가지는지 파악하는데도 유용하게 사용된다.

corr()함수를 적용하여 변수간 상관관계 분석 후, 상관관계가 0.3이상인 변수만 출력하였다.
```python
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) # 한글 출력 설정 부분

cancer_tmp = cancer.copy()
cancer_tmp['diagnosis'] = cancer['diagnosis'].replace({'M':1, 'B':0})
corrmat = cancer_tmp.corr()
top_corr_features = corrmat.index[abs(corrmat["diagnosis"])>=0.3]

# plot
plt.figure(figsize=(13,10))
g = sns.heatmap(cancer[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
```
<img src="/image/cancer_correaltion.png" width="750" height="400">
<br/>


## 5. Feature Importances
머신러닝 및 딥러닝 예측 후 어떻게 이러한 결과가 나왔는지 의문이 들 때가 있다. 그럴 땐 변수 중요도를 통해 어떤 변수가 예측 성능에 주요하게 영향을 미쳤는지 파악할 수 있다. 

RandomForest 알고리즘을 통해 feature importances를 뽑아낸 후 상위 중요도 별로 중요도가 0이상만 출력하였다.
```python
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) # 한글 출력 설정 부분

# RandomForest
clf = RandomForestClassifier(random_state=42, max_depth=6)
clf.fit(X, y)
feature_importance = clf.feature_importances_

# plot
df_fi = pd.DataFrame({'columns':X.columns, 'importances':feature_importance})
df_fi = df_fi[df_fi['importances'] > 0] # importance가 0이상인 것만 
df_fi = df_fi.sort_values(by=['importances'], ascending=False)

fig = plt.figure(figsize=(15,7))
ax = sns.barplot(df_fi['columns'], df_fi['importances'])
ax.set_xticklabels(df_fi['columns'], rotation=80, fontsize=13)
plt.tight_layout()
plt.show()
```
<img src="/image/cancer_fi.png" width="1000" height="400">
변수 중요도 출력결과 concave_points_worst가 0.175로 가장 중요한 예측 변수이며, 그 뒤로 perimeter_worst, perimeter_mean, radius_word가 주요 예측 변수로 나타났다.

---
Outro
---
이번에는 기초적인 데이터 시각화를 알아보았는데, 데이터 시각화는 데이터와 상황에 따라 그때 그때 시각화해야하는 요소가 다르고 다양하기 때문에, 상황에 따라 원하는 그림을 그리며 자신만의 인사이트를 찾아나가면 될 듯 하다.