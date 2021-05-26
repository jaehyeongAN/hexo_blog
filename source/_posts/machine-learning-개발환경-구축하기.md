---
title: 머신러닝을 위한 아나콘다(Anaconda) 개발환경 구축
date: 2019-04-09 22:06:02
tags:
- python
- anaconda
- miniconda
- jupyter
- sublimetext3
- datascience
- machinelearning
---
Intro
---
대부분 처음 머신러닝를 하고자 마음먹고 시도하는 것이 개발환경을 만드는 것 입니다.
물론 데이터 사이언스에 대한 기본적 이해가 먼저 뒷받침이 되어 있어야 하겠지만 처음부터 너무 어렵게 시작하면 재미없자나요? 먼저 python의 세계에서 hello world 부터 찍어봐야져.

오늘은 Machine learning, data science 뭐 등등을 하기 위해 가장 많이 사용되고 있는 언어인 Python 개발환경을 구축하는 방법을 소개합니다.(내 개발환경 기준으로)
(JDK는 설치되어있어야 합니다.)

---
굳이 python을 설치할 필요 없이, data science 패키지 모음인 Anaconda를 설치할 것입니다.
Anaconda는 수학 및 과학 관련 numpy, scipy, pandas, matplotlib과 같은 유용한 python package를 모아놓은 배포판입니다. 분명히 장점이긴 하지만 잘 사용하지 않는 패키지까지 모두 포함하고 있어 무겁다는 단점이 있습니다.

따라서, 저는 Anaconda의 축소판인 Miniconda를 깔도록 하겠습니다. 
Miniconda는 Anaconda와 다르게 사용하고자 하는 패키지를 스스로 설치해야하는 번거로움이 있지만 가볍다는 장점이 있습니다.

##### Anaconda vs Miniconda
Anaconda
-- python이나 conda를 처음 접하는 경우 좋음
-- python과 150개 이상의 과학 패키지를 한 번에 자동 설치하여 편리
-- 강력한 script editor인 Jupyter notebook이 포함되어 있음
-- 3gb의 여유 용량이 필요 

Miniconda
-- 적은 용량(100mb 이하)
-- 스스로 원하는 패키지를 설치하고자 할 경우 좋음(좋은 습관)
<p></p>

Install Miniconda(+python)
---
우선 Miniconda 홈페이지를 가면 설치가 가능합니다.
<img src="/image/miniconda-ori.PNG" width="1000">

~~위 링크에서는 현재 python 2.7버전과 3.7버전을 제공하고 있습니다. 최신버전이므로 다운받으셔도 무방하지만 한 가지 고려할 것이 있습니다. 만약 추후 deep learning을 하고자 한다면 3.7버전 보다 아래 버전을 사용하시는 것이 좋습니다. tensorflow가 아직 3.7버전에 호환되지 않거든여ㅜㅜ~~
~~그래서 [Miniconda installer archive](https://repo.continuum.io/miniconda/)에서 python3.6버전으로 되어있는 것을 찾으시면 됩니다.저는 안정성 문제를 고려하여 [Miniconda2-4.5.4-Windows-x86_64.exe](https://repo.continuum.io/miniconda/Miniconda2-4.5.4-Windows-x86_64.exe)을 이용하고 있습니다.~~

(수정)다시 알아보니 이제 python 3.7버전도 tensorflow와 호환이 된다네요! 그냥 [Miniconda 홈페이지](https://docs.conda.io/en/latest/miniconda.html)에서 최신버전을 다운받으셔도 무방할 것 같습니다.


다운받은 후 설치파일을 더블클릭하면 아래와 같은 화면이 나타납니다.
<img src="/image/miniconda-install.PNG" width="600">

이후, next버튼을 눌러 설치를 진행하시면 됩니다. 

설치가 완료되면 설치된 경로에 아래와 같이 miniconda가 깔립니다.(동시에 python도 같은 경로에 설치가 됩니다.)
<img src="/image/miniconda-path.PNG" width="600">
<p></p>

Enroll System path
---
자, 이제 miniconda 및 python을 설치하였으니 환경변수에 등록해줘야 합니다. java jdk 등록하는거와 같습니다.
우선 환경변수에 등록해야 하는 path는 아래와 같습니다.(본인이 설치한 path에 맞게 넣어주시면 됩니다.)
<img src="/image/miniconda-syspath.PNG" width="500">

- C:\Users\nonam\Miniconda3
- C:\Users\nonam\Miniconda3\python.exe
- C:\Users\nonam\Miniconda3\Scripts
- C:\Users\nonam\Miniconda3\Library\bin

anaconda 및 miniconda를 설치하게 되면 anaconda prompt와 같은 콘솔창이 함께 설치됩니다. 
검색창에서 anaconda prompt를 실행하여 아래와 같이 python이 실행된다면 설치가 완료된 것입니다.
<img src="/image/anaconda-prompt.PNG" width="800">
<p></p>

Install Jupyter lab
---
이제 강력한 data science 에디터인 jupyter lab을 설치해보겠습니다.(jupyter lab은 jupyter notebook보다 다양한 기능을 가지고 있고 파일 관리가 쉬워 애용합니다.)
anaconda prompt를 열어 아래와 같이 명령어(jupyter lab 및 해당 kernel을 설치)를 실행합니다. 
```console 
> conda install -c conda-forge jupyterlab

> python -m ipykernel install --user
```

실행이 되면 http://localhost:8888/ 주소로 jupyter lab이 실행되며 아래와 같은 화면이 나타납니다.
<img src="/image/jupyter.PNG" width="900">
<p></p>

Outro
---
jupyter notebook은 communication computing shell이라고 할 수 있습니다. python의 결과를 바로바로 확인가능하기 때문에 데이터 분석 및 시각화에서 아주 강력한 툴이지요.
하지만, 대용량 데이터를 처리해야 하고 loop및 조건문이 자주 코드에 포함된다면 조금 다를 수 있습니다.
jupyter notebook은 변수에 메모리를 적재 후 지속적으로 메모리를 차지하기 때문에 대용량 데이터와 같은 고성능 데이터 처리에는 그다지 추천드리지 않습니다. 

따라서 저는 간단히 데이터의 분포 및 분석을 위해서만 jupyter를 사용하는 편입니다. 그 외 전체적인 코딩은 [sublime text3](https://www.sublimetext.com/3)라는 editor를 사용하고 있습니다.
추후 python 코딩을 위한 강력한 또다른 editor인 sublime text3에 대해 소개해드리도록 하겠습니다. 