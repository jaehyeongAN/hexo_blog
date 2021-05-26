---
title: AWS EC2에 플라스크(Flask) 클라우드 웹 서버 구축하기
date: 2020-01-13 21:26:58
tags:
- flask
- python
- web
- amazon
- aws
- webservice
- cloud
- webframework
---
Intro 
---
지난 번 글에서 Flask 웹 프레임워크를 통해 간단한 딥러닝 웹 애플리케이션을 개발해보았다. 하지만 로컬(local) 환경에서 개발하였기 때문에 개발 서버를 종일 켜놓거나 고정 도메인을 따로 받지 않은 이상 외부 IP로 접근은 불가능하다. 
그렇기 때문에 나처럼 물리적인 서버를 구축 및 운영할 환경이 되지 않을 경우는 클라우드(Cloud) 서비스를 이용하게 되는데, 

이번 글에서는 **AWS(Amazon Web Services)**라고 하는 클라우드 서비스를 활용하여 웹 서버를 구축 후 Flask를 배포하는 과정을 설명하려고 한다.

---
## 1. AWS EC2 가입 및 인스턴스 생성 
우선 [AWS Management Consol](https://ap-northeast-2.console.aws.amazon.com/console/home?region=ap-northeast-2#)로 이동 후 가입이 되어있지 않다면 가입 후 로그인을 한다. (가입 시 region을 Seoul로 설정할 것)
서비스 검색을 통해 EC2를 선택한다. 
<img src="/image/aws-mc.png" width="1000"/>

EC2 대시보드에서 인스턴스 생성 아래의 **인스턴스 시작** 버튼을 클릭
<img src="/image/aws-instance.png" width="1000"/>

AMI로는 기업용이 아니니 개인 개발용으로 편한 **Ubuntu Linux 18.04** 버전을 사용하며,
무료 서버 이용이 가능한 **프리 티어(Free Tier)**로 서버를 생성한다. 
<img src="/image/aws-ami.png" width="1000"/>
<img src="/image/aws-free.png" width="1000"/>
<img src="/image/aws-start.png" width="1000"/>

위 이미지에서 시작 버튼을 누를 경우 키 페어를 설정하는 메시지가 나타나는데 이 키 페어는 말 그대로 생성한 웹 서버에 추후 접속할 때 꼭 필요한 키 역할을 한다. **'새 키 페어 생성'**을 선택하고 **'키 페어 이름'**을 본인 취향에 맞게 설정 후 **'키 페어 다운로드'**를 선택한다. (이 키 페어는 추후 서버 접속 시 꼭 필요하므로 본인 개발 폴더에 잘 보관해둔다.)
키 페어를 다운로드하여 인스턴스 시작 버튼이 활성화되면 버튼을 클릭하여 진행한다. 
<img src="/image/aws-keypair.png" width="1000"/>

**인스턴스 보기**를 선택
<img src="/image/aws-status.png" width="1000"/>

여기까지가 진행하게 되면 인스턴스가 아래와 같이 생성된다. 
<img src="/image/aws-instance-view.png" width="1000"/>
<br>

## 2. Key Pair 권한 설정 변경
전 과정에서 인스턴스를 생성하면서 Key Pair를 같이 다운로드 하였을 것이다. 하지만 접속하기 위해서는 이 권한 설정을 변경해줘야만 접속이 가능하다. 

우선 다운받은 키페어를 우클릭하여 **[속성]-[보안]** 탭으로 이동 후 **[고급]**을 클릭한다. 
<img src="/image/aws-admin.png" width="700"/>

아래와 같은 화면에서,
**[상속 사용 안 함]**을 클릭 후, 팝업 메시지에서 **'상속된 사용 권한을 이 개체에 대한 명시적 사용 권한으로 변환합니다'**를 선택
<img src="/image/aws-admin-remove1.png" width="700"/>
<img src="/image/aws-admin-remove2.png" width="700"/>

이후 아래와 같이 Administrators를 제외한 모든 사용 권한 항목을 제거한다
<img src="/image/aws-admin-remove3.png" width="700"/>
<br>

## 3. 보안 그룹 설정
인스턴스 화면으로 돌아와 **Flask 웹 서버 포트 번호인 5000번 포트**를 열어 주기 위해 보안 그룹을 설정한다. 
<img src="/image/aws-security.png" width="1000"/>

[인바운드] 탭에서 [편집]을 클릭 후 [규칙 추가]를 하여 아래와 같이 5000번 포트를 설정한다.
<img src="/image/aws-security2.png" width="1000"/>

## 4. 인스턴스 접속하기
다시 인스턴스 화면으로 돌아와 아래 화면에서 생성한 인스턴스를 선택 후 연결 버튼을 클릭한다. 
연결 방법으로는 **'독립 실행형 SSH 클라이언트'**로 선택하고, 아래 ssh 명령어를 복사한다. 
<img src="/image/aws-connect.png" width="1000"/>
<img src="/image/aws-connect2.png" width="1000"/>

명령프롬프트(CMD)를 관리자 권한으로 실행 후 다운 받은 Key Pair가 있는 위치로 이동한다. 그 후 위에서 복사한 SSH 명령어를 복사하여 우분투 리눅스 인스턴스에 접속한다. 
<img src="/image/aws-ubuntu.png" width="800"/>

우선 파이썬 라이브러리 도구인 pip 및 java jdk 등을 설치해주고, 본인의 파이썬 코드가 수행되기 위한 라이브러리를 설치해준다. 
<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div><div style="line-height:130%">7</div><div style="line-height:130%">8</div><div style="line-height:130%">9</div><div style="line-height:130%">10</div><div style="line-height:130%">11</div><div style="line-height:130%">12</div><div style="line-height:130%">13</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;update</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#999999">#&nbsp;java&nbsp;설치&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;openjdk<span style="color:#0086b3"></span><span style="color:#ff3399">-</span><span style="color:#c10aff">8</span><span style="color:#ff3399">-</span>jre</div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;openjdk<span style="color:#0086b3"></span><span style="color:#ff3399">-</span><span style="color:#c10aff">8</span><span style="color:#ff3399">-</span>jdk</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#999999">#&nbsp;pip&nbsp;설치&nbsp;및&nbsp;라이브러리&nbsp;설치</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;python3<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>pip</div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;tensorflow</div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;keras</div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;opencv<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>python</div><div style="padding:0 6px; white-space:pre; line-height:130%">$&nbsp;sudo&nbsp;apt&nbsp;install&nbsp;scipy</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</div></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>


이후 개발한 flask를 웹 서버로 clone하여 해당 경로로 이동 후 웹 서버를 실행해준다.
<img src="/image/aws-ubuntu2.png" width="650"/>
<img src="/image/aws-ubuntu3.png" width="650"/>

이제 웹 서버가 실행 중이니 퍼블릭 IP로 접속이 가능하다. 아래 인스턴스 화면에서 **'IPv4 퍼블릭 IP'** 주소를 복사 후 5000번 포트번호( http://54.180.150.154:5000/ )로 접속한다.

<img src="/image/aws-dl-flask.PNG" width="1000"/>
고정 IP에서 서버가 잘 실행되고 있다.
<br>

## 5. 파이썬 서버 계속 실행 시키기
위와 같이 정상적으로 고정 IP를 통해 접속이 가능함을 확인하였다. 하지만 SSH 프롬프트를 종료하게 되면 파이썬 서버도 함께 종료되게 된다. 마지막으로 파이썬 서버가 항상 실행될 수 있도록 설정한다. 

1. Ctrl+Z 를 통해 파이썬 프로세스 중지
2. $ bg : 백그라운드에서 프로세스 재 구동
3. $ disown -h : 소유권 포기 


---
### References
- https://ndb796.tistory.com/244