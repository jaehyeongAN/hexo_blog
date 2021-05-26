---
title: 파이썬 소켓(socket) 프로그래밍
date: 2018-06-29 17:22:55
tags:
- python
- network
- server
- client
- socket
---
# 파이썬 소켓(socket) 프로그래밍
------
소켓(socket)을 통해 서버(server)와 클라이언트(client)간 어떻게 기본적인 네트워크 통신이 이루어지는지 알아보려고 합니다.
![check](https://www.tutorialspoint.com/perl/images/perl_socket.jpg)
먼저 통신을 위해 두개의 파일은 준비합니다. 파일은 각각 서버와 클라이언트에 해당합니다.

- server.py
- client.py

우선 **server.py** 작성

```python
from socket import *
from select import *

HOST = ''
PORT = 10000
BUFSIZE = 1024
ADDR = (HOST, PORT)

# 소켓 생성
serverSocket = socket(AF_INET, SOCK_STREAM)

# 소켓 주소 정보 할당 
serverSocket.bind(ADDR)
print('bind')

# 연결 수신 대기 상태
serverSocket.listen(100)
print('listen')

# 연결 수락
clientSocekt, addr_info = serverSocket.accept()
print('accept')
print('--client information--')
print(clientSocekt)

# 클라이언트로부터 메시지를 가져옴
data = clientSocekt.recv(65535)
print('recieve data : ',data.decode())

# 소켓 종료 
clientSocekt.close()
serverSocket.close()
print('close')
```

1. 우선 소켓을 설정하고, bind()함수를 통해 주소 정보를 할당한다.
2. 이후, listen()함수를 통해 연결 수신 대기 상태로 전환 후 
3. client가 연결할 시 accpet() 함수를 이용하여 연결을 수락한다.
4. 만약 client가 보낸 메시지가 있을 경우, recv(byte크기)를 이용하여 메시지를 가져온다.



이제 **client.py**를 작성

```python
#! /usr/bin/python
# -*- coding: utf-8 -*-

from socket import *
from select import *
import sys
from time import ctime

HOST = '127.0.0.1'
PORT = 10000
BUFSIZE = 1024
ADDR = (HOST,PORT)

clientSocket = socket(AF_INET, SOCK_STREAM)# 서버에 접속하기 위한 소켓을 생성한다.

try:
	clientSocket.connect(ADDR)# 서버에 접속을 시도한다.
	clientSocket.send('Hello!'.encode())	# 서버에 메시지 전달

except  Exception as e:
    print('%s:%s'%ADDR)
    sys.exit()

print('connect is success')
```

1. 주소와 포트번호를 설정
2. server에 접속하기 위한 client socket을 생성하고
3. connect()함수를 이용하여 서버에 접속을 시도
4. send()함수를 이용해 메시지를 server에 전달



------

이제 위의 코드를 실행해보도록 하겠습니다. 

server.py를 실행 후 client.py를 통해 server에 접속하는 과정입니다.



1. 먼저 server.py를 실행하여, client의 접속을 기다립니다.

   ![server.py](/image/server.png)

   

2. 이후,  client.py를 실행하여 server에 접속을 시도합니다.

   ![client.py](/image/client.png)



3. server에서 client의 접속정보와 메시지를 확인합니다.

   ![check](/image/check.png)