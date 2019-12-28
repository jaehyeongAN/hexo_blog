---
title: 파이썬 웹 프레임워크 Flask를 활용한 딥러닝 웹 애플리케이션 개발 
date: 2019-12-27 23:20:58
tags:
- python
- flask
- deeplearning
- webdevelopment
- webframework
- django
- neuralstyletransfer
---
Intro
---
Java의 Spring처럼 Python에서도 웹 프레임워크를 제공한다. 그 중 가장 인기 있는 것이 Django와 Flask인데, Django의 경우 Instagram, LinkedIn 사이트로 사용될 정도로 인기 있고 안정적인 웹 프레임워크라고 할 수 있다. 그 만큼 체계적이고 정교한 구조를 가지고 있다고 할 수 있는데 그와 반대로 Flask는 좀 더 간편하고 경량화 된 웹 프레임워크라고 생각하면 될 것 같다. 그래서 실제 서비스 하기 보다는 간단한 프로토타입 개발 용도로 많이 사용되는 것 같다. 

이번에는 Flask 웹 프레임워크에 대해 알아보고 딥러닝 모델 중 Neural Style Transfer를 Flask에서 실행하여 결과를 웹으로 표출해보도록 할 것이다.

>>> *※ 해당 전체 코드는 [github](https://github.com/jaehyeongAN/PyFlask_DL-service)에서 확인 할 수 있습니다.*

---
## 1. Flask 설치 

#### 1) flask 프로젝트 폴더 생성 
우선 flask 프로젝트를 수행할 폴더를 본인의 임의 경로에 설치해준다. 나는 아래와 같은 경로에 폴더를 만들었다.
```
cd workspace
workspace > mkdir pyflask
```
<img src="/image/flask_dir.JPG" width="500" height="200">

#### 2) 가상환경
flask 환경을 위한 virtualenv 가상환경 라이브러리를 설치한다.
```
pip install virtualenv
```

설치 완료 후 본인의 flask 경로 내에서 가상환경을 생성해준다. 
```
workspace\pyflask > virtualenv venv
```

정상적으로 실행 시 flask 폴더 내에 venv 폴더가 생성되는데 venv/Scripts 폴더로 이동하여 가상환경을 활성화(active) 한다.
```
workspace\pyflask > cd venv/Scripts
workspace\pyflask\venv\Scripts > activate
```

#### 3) Flask 설치 
위에서 active한 가상환경 내에서 flask를 설치해준다. 
```
(venv) pip install flask
```

설치완료 된 flask의 버전은 아래처럼 확인할 수 있다.
```
flask --version
```
<img src="/image/flask_version.JPG" width="400" height="200">

<br>

## 2. 웹 구성
이제 Flask를 개발할 환경은 구축하였으니, 웹 애플리케이션의 구조를 설계해 볼 것이다.
여기서 프로토타입으로 구성해 볼 웬 애플리케이션은 간단하게 메인 페이지로 구성되고, 각 기능을 수행하는 서브 페이지로 이동하게 된다. 이후 사용자로 부터 입력값을 받아 딥러닝 기능 수행 후 그 결과값을 다시 웹으로 출력해주는 구조를 가진다. 
```
index 		(메인 페이지)
├── nst_get 	(user input 받는 페이지)
└── nst_post	(결과 출력 페이지)
```
<br>
#### 1) 폴더 구성
우선 효율적인 웹 개발을 위하여 flask 폴더 내에 몇 개의 폴더를 설치하여 기능별로 관리한다. 
```
pyflask/
├── static/
	└── images/
├── templates/
├── venv/
└── neural_style_transfer.py
```

- static/images : 사용자로부터 받을 이미지를 저장할 경로
- templates : html 파일 
- neural_style_transfer.py : neural style transfer를 수행할 딥러닝 코드

<br>
#### 2) HTML 템플릿
우선 화면 구성을 위하여 HTML 템플릿을 아래와 같이 최대한 간단하게 작성하였다. 
(지면상 CSS와 JS코드는 제거하였는데 전체 코드는 [github](https://github.com/jaehyeongAN/PyFlask_DL-service)에서 확인할 수 있다.)

 * index.html (메인 페이지)
 ```html
<!DOCTYPE html>
<html lang="ko">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">
	<title>Flask Index</title>
</head>

<body>
	<br>
	<h1 align="center">Flask for Deep ConvNet</h1>
	<br>
	<ul>
		<li><h2><a href="/nst_get">Neural Style Transfer</a></h2></li>
		<li><h2><a href="#">Obejct Detection</a></h2></li>
	</ul>
</body>
<footer align='center'>Powerd by <strong>© 2019 JaeHyeong</strong></footer>
</html>
 ```
 -- a태그 링크에 /nst_get을 명시하여 클릭 시 nst_get.html로 이동
 <br>

 * nst_get.html (user input 받는 페이지)
  ```html
<!DOCTYPE html>
<html lang="ko">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">
	<title>Flask image get</title>
</head>
<body>	
	<br>
	<h1 align="center">Neural Sytle Transfer</h1>
	<br>
	<form align="center" action="/nst_post" method="POST" enctype="multipart/form-data">
		<h2 align="center">Reference Images</h2>
		<table align="center">
			<tr>
				<td><img class="refer_img" id="refer_img1" src="./static/images/rain_princess.jpg"></td>
				<td><img class="refer_img" id="refer_img2" src="./static/images/the_stary_night.JPG"></td>
				<td><img class="refer_img" id="refer_img3" src="./static/images/scream.jpg"></td>
				<td><img class="refer_img" id="refer_img3" src="./static/images/zentangle_art.jpg"></td>
			</tr>
			<tr>
				<td><input type="radio" name="refer_img" value="rain_princess.jpg"></td>
				<td><input type="radio" name="refer_img" value="the_stary_night.JPG"></td>
				<td><input type="radio" name="refer_img" value="scream.jpg"></td>
				<td><input type="radio" name="refer_img" value="zentangle_art.jpg"></td>
			</tr>

		</table>
		<br><br>
		<h2 align="center">Target Image</h2>
		<div align="center" id='view_area'></div>
		<br>
		<input type="file" name="user_img" id="user_img" value="userIMgage" onchange="previewImage(this,'view_area')"/>
		<input type="submit" value="확인"/>
	</form>
	<br><br><br>

</body>
</html> 
 ```
 -- 이미지를 Flask로 넘겨주는 페이지이므로 form 태그의 POST 방식을 수행 
 -- neural style transfer 학습을 위한 reference 이미지 경로를 지정하여 표시
 -- 사용자로부터 이미지 파일을 입력 받음
 -- 확인 버튼을 통해 선택한 reference 이미지와 사용자 이미지를 전송 
 <br>

 * nst_post.html (결과 출력 페이지)
 ```html
<!DOCTYPE html>
<html lang="ko">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<meta http-equiv="X-UA-Compatible" content="ie=edge">
	<title>Flask image post</title>
</head>
<body>
	<br>
	<table align="center">
		<tr>
			<td><h2 align="center">Reference Image</h2></td>
			<td><h2 align="center">Target Image</h2></td>
		</tr>
		<tr>
			<td><img class='nst_img' src="{{url_for('static', filename=refer_img)}}"></td>
			<td><img class=nst_img src="{{url_for('static', filename=user_img)}}"></td>
		</tr>
		<tr>
			<td colspan="2"><h2 align="center">Transfer Image</h2></td>
		</tr>
		<tr>
			<td colspan="2" align="center"><img class="nst_result_img" src="{{url_for('static', filename=transfer_img)}}"></td>
		</tr>
	</table>
	<br>
</body>
</html>
 ```
 -- Flask로 부터 넘겨 받은 결과 이미지를 받아서 출력 

<br>
#### 3) Neural Style Transfer 수행 코드 작성 
전체 코드는 [github](https://github.com/jaehyeongAN/PyFlask_DL-service/blob/master/flask_deep/neural_style_transfer.py) 참조 

```python
def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_height, img_width)) # (400, 381)
	img = img_to_array(img) 			# (400, 381, 3)
	img = np.expand_dims(img, axis=0) 	# (1, 400, 381, 3)
	img = vgg19.preprocess_input(img)
	return img 

				:
				: ( 중략 )
				:

def main(refer_img_path, target_img_path):
	style_reference_image_path = 'flask_deep/static/'+ refer_img_path
	target_image_path = 'flask_deep/static/'+ target_img_path

	# 모든 이미지를 fixed-size(400pixel)로 변경
	width, height = load_img(target_image_path).size
	global img_height; global img_width;
	img_height = 400
	img_width = int(width * img_height / height)

	target_image = K.constant(preprocess_image(target_image_path)) # creates img to a constant tensor
	style_reference_image = K.constant(preprocess_image(style_reference_image_path))
	combination_image = K.placeholder((1, img_height, img_width, 3)) # 생성된 이미지를 담을 placeholder

	# 3개의 이미지를 하나의 배치로 합침
	input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

	# 3개 이미지의 배치를 입력으로 받는 VGGNet 생성
	model = vgg19.VGG19(input_tensor=input_tensor, 
						weights='imagenet', # pre-trained ImageNet 가중치 로드 
						include_top=False) # FC layer 제외 

				:
				: ( 중략 )
				:

	evaluator = Evaluator()
	refer_img_name = refer_img_path.split('.')[0].split('/')[-1]
	result_prefix = 'flask_deep/static/images/nst_result_'+refer_img_name
	iterations = 30

	# 뉴럴 스타일 트랜스퍼의 손실을 최소화하기 위해 생성된 이미지에 대해 L-BFGS 최적화를 수행
	x = preprocess_image(target_image_path)
	x = x.flatten()
	for i in range(iterations):
		start_time = time.time()
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
										 fprime=evaluator.grads, maxfun=20)
		# 생성된 현재 이미지를 저장
		img = x.copy().reshape((img_height, img_width, 3))
		img = deprocess_image(img)
		fname = result_prefix + '.png'
		end_time = time.time()

	save_img(fname, img)


	return fname

if __name__ == "__main__":
	main()
```

<br>

#### 4) Flask app 파일 생성 
Flask 애플리케이션 실행을 위한 \__init\__.py 파일을 생성한다. 여기서 Flask 파라미터로 전달되는 \__name\__ 파라미터는 Flask 애플리케이션을 구분하기 위한 구분자로 사용된다.
app.debug 를 True로 지정할 경우 코드 수정 시 바로바로 디버깅이 가능하게 해준다.

@app.route는 페이지 URL과 함수를 연결해주는 역할을 하며 아래와 같이 @app.route 데코레이터 지정 후 render_template('URL')을 통해 연결할 페이지 경로를 입력하면 해당 경로를 웹 브라우저로 전달해주게 된다. 

* \__init\__.py
```python
import os, sys
from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from werkzeug import secure_filename
from . import neural_style_transfer

app = Flask(__name__)
app.debug = True

# Main page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/nst_get')
def nst_get():
	return render_template('nst_get.html')

@app.route('/nst_post', methods=['GET','POST'])
def nst_post():
	if request.method == 'POST':
		# Reference Image
		refer_img = request.form['refer_img']
		refer_img_path = 'static/images/'+str(refer_img)

		# User Image (target image)
		user_img = request.files['user_img']
		user_img.save('./flask_deep/static/images/'+str(user_img.filename))
		user_img_path = './static/images/'+str(user_img.filename)

		# Neural Style Transfer 
		transfer_img = neural_style_transfer.main(refer_img_path, user_img_path)
		transfer_img_path = './static/images/'+str(transfer_img.split('/')[-1])

	return render_template('nst_post.html', 
					refer_img=refer_img_path, user_img=user_img_path, transfer_img=transfer_img_path)

```
 -- index에서 nst_get 링크 클릭 시 nst_get.html로 경로 이동
 -- nst_get.html에서 POST방식을 통해 전달받은 이미지를 nst_post 함수에서 request 메소드를 통해 넘겨받음
 -- reference 이미지와 user 이미지 경로를 neural style transfer를 수행하는 딥러닝 메소드로 전달
 -- neural style transfer 딥러닝 코드에서 모델 수행 후 받은 결과값을 nst_post.html로 전달 

<br>

## 3. 웹 실행
#### 1) Flask 서버 실행
Flask 서버 실행을 위해 프로젝트 폴더 상위에 아래와 같은 파이썬 코드를 생성하였다.
```python
from pyflask import app

app.run(host='127.0.0.1')
```
위 코드는 \__init\__.py에서 app 애플리케이션을 실행하게 해주며, 위 파이썬 파일 실행 시 아래와 같이 flask 서버가 실행되게 된다. 이후 브라우저에서 http://127.0.0.1:5000 입력 시 위에서 만든 화면을 확인할 수 있다.

<img src="/image/flask_run.JPG" width="650" height="200">

#### 2) 웹 화면 
* 메인 페이지
<img src="/image/flask_index.JPG" width="1000" height="200">

* 사용자 입력 페이지
<img src="/image/flask_get.JPG" width="1000" height="200">

* 결과 출력 페이지
<img src="/image/flask_result.JPG" width="1000" height="200">

Outro
---
Flask는 이번에 간단한 애플리케이션을 적용해보기 위해서 처음 사용해보았다. 이전에 Spring 프레임워크를 통해 프로젝트를 해본 적은 있는데 사용안하지 너무 오래되다보니 까먹기도 했고 환경 구성하는 것도 일이어서 좀 가벼운 Flask를 사용해보게 되었다.

일단 기본적으로 html/css 그리고 python만 기초적으로 알아도 누구나 쉽고 간단하게 웹 화면을 구성해볼 수 있다는 장점이 있는 것 같다. 근데 확실히 큰 프로젝트 성으로 여러사람이 복잡한 화면을 구성할 때는 작업 관리가 쉽지 않을 것 같다는 생각이 든다. 시간나면 Django도 공부해봐야할 것 같다. 