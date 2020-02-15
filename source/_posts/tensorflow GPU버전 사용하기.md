---
title: 텐서플로우(Tensorflow 2.0) GPU버전 사용하기
date: 2019-05-01 18:29:10
tags:
- python
- tensorflow
- keras
- nvidia
- cuda
- cudnn
---
Intro
---
머신러닝 모델을 학습할때는 크게 와닿지 않지만 복잡한 딥러닝 연산을 할 때 크게 느껴지는 것이 바로 GPU의 유무이다. 딥러닝과 같은 복잡한 matrix 연산을 하기 위해 CPU로 모델을 돌렸다가는 컴퓨터가 운명을 다 할 수 있다. 
이전에 텍스트 처리 딥러닝 모델을 CPU와 GPU로 돌렸을 때 얼마나 차이나는지 보려고 실험을 했었는데, GPU의 경우 3시간 정도만에 학습이 끝난 반면 CPU의 경우 거의 한나절을 돌아가고도 결과가 나오지 않아 중간에 끊은 적이 있었다. 

본인 컴퓨터에 외장 그래픽이 없다면 할 수 없지만 GPU가 갖춰져 있을 경우 이를 적극 활용하는 것이 정신건강에 좋을 것 같다. 
하지만, GPU도 다 같은 GPU가 아니다.
현재 tensorflow에서 지원하는 GPU는 Nvidia를 기본으로 하며 AMD의 경우 아직 이용하기에 많이 불편하다. 

---

tensorflow-gpu 설치
---
#### 1. CUDA 설치 
우선 CUDA를 설치해야 한다. 현재 CUDA의 경우 최신 버전이 10.2이지만, 확인 결과 아직까지는 공식적으로 tensorflow가 CUDA 10.0버전까지만 지원한다. 

[CUDA Toolkit Arcive(https://developer.nvidia.com/cuda-toolkit-archive)](https://developer.nvidia.com/cuda-toolkit-archive)로 이동하여 아래 화면과 같이 **CUDA Toolkit 10.0**버전을 클릭한다. 

<img src="/image/cuda_toolkit.JPG" width="1000" height="400">

<br />
클릭 후 아래와 같이 자신의 운영체제 맞는 것을 선택한 후 다운로드를 실시하고 다운로드 된 설치파일을 다른 조건 변경없이 그대로 설치하면 된다. 

<img src="/image/cuda_toolkit2.JPG" width="1000" height="400">


#### 2. cuDNN 다운로드 
CUDA 설치를 완료하였다면 이제 [cuDNN(https://developer.nvidia.com/rdp/cudnn-download)](https://developer.nvidia.com/rdp/cudnn-download)을 다운로드하여 CUDA 디렉토리에 넣어줘야 한다. 
cuDNN을 설치하기 위해서는 nvidia에 로그인을 해야하므로 가입이 안되어있다면 가입을 한 후 접속하면 된다. 

주의할 점은 위에서 설치한 CUDA버전에 호환되는 cuDNN을 다운로드 해야 한다는 것이다. 위에서 CUDA 10.0버전을 설치해주었기 때문에 cuDNN도 CUDA 10.0에 호환되는 버전(for CUDA 10.0)으로 다운받는다.

<img src="/image/cudnn.JPG" width="1000" height="400">

위 파일을 다운로드 하면 **cudnn-10.0-windows10-x64-v7.5.1.10** 라는 압축파일이 다운로드 되는데, 압축파일을 풀게 되면 그 안에 아래와 같은 파일이 들어있다.

<img src="/image/cudnn2.JPG" width="800" height="400">
<br />

#### 3. cuDNN파일 CUDA 폴더로 복사
이제부터가 중요한데, 
방금 전 압축해제 한 폴더의 파일을 모두 복사하여 그대로 처음 설치한 CUDA 폴더로 전부 복사해주어야 한다. 
우선 압축해제 한 파일들을 전부 복사한 후, **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0** 이 경로로 가서 복사한 파일을 그대로 붙여 넣기 해준다.(안에 동일한 파일이 있는데 그냥 덮어씌워주는 것이다.)

<img src="/image/cudnn3.JPG" width="800" height="400">
(위 경로에 그대로 복사한 파일을 덮어씌운다.)
<br />


#### 4. 환경변수 지정 
보통 다른 설정을 건드리지 않고 진행하였을 경우, 환경변수에 아래와 같은 CUDA 경로가 들어있을 것이다. 없다면 아래와 같은 경로를 그대로 환경변수에 지정해준다. 
<img src="/image/path.JPG" width="500" height="400">
<br />

#### 5. tensorflow-gpu 버전 설치 
이후 Anacoda prompt 혹은 CMD 창을 열어 아래와 같은 명령어로 tensorflow-gpu버전을 설치한다.

**> pip install tensorflow-gpu**
혹은 
**> conda install tensorflow-gpu**

<img src="/image/install_tensorflow_gpu.JPG" width="700" height="400">
(이미 설치되어 있어서 위와 같이 나옴.)
<br />

#### 6. tensorflow 실행 및 확인 
promt창을 열어 아래와 같이 tensorflow를 import하였을 때  error가 나지 않는다면 우선 tensorflow 설치에 성공한 것이다. 
설치 된 tensorflow 버전을 확인하고 싶을 때는 tf.\__version\__ 을 통해 확인할 수 있다.
```python
import tensorflow as tf 
tf.__version__
```
<img src="/image/tensorflow-version.JPG" width="900" height="400">

tensorflow가 GPU버전으로 잘 설치되었고, 나의 GPU를 잘 인식하고 있는지 확인하고 싶다면 아래와 같은 코드를 통해 확인할 수 있다. tensorflow가 인식하는 로컬 device 목록을 보여주게 된다.
```python
from tensorflow.pyhton.client import device_lib
device_lib.list_local_devices()
```

<img src="/image/check-tensorflow-gpu.JPG" width="1100" height="400">

내 컴퓨터의 GPU의 경우 GeForce GTX 1050 with MAX-Q인 것을 확인할 수 있다. 

