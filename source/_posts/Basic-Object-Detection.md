---
title: Basic Object-Detection
date: 2020-04-15 18:34:17
tags:
- objectdetection
- region-proposal
- selectivesearch
- IoU
- NMS
- mAP
---
Intro
---
*Inflearn의 [딥러닝 컴퓨터 비전 완벽 가이드](https://www.inflearn.com/course/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%BB%B4%ED%93%A8%ED%84%B0%EB%B9%84%EC%A0%84-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C/dashboard)를 수강하며 공부 목적으로 정리한 글입니다.*

---
## Computer Vision Techniques 
<img src="/image/computer-vision-problem.png" width="900px">
- Classification(분류) : 이미지에 있는 object가 무엇인지만 판별, 위치 고려 x
- Localization(발견) : object 판별 및 단 하나의 object 위치를 bounding box로 지정하여 찾음
- Detection(발견) : object 판별 및 여러 개의 object들에 대한 위치를 bounding box로 지정하여 찾음
- Segmentation(분할) : object 판별 및 Pixel 레벨의 detection을 통해 모든 픽셀의 레이블을 예측
<br>

## Object Detection
### History
<img src="/image/object-detection-history.png" width="900px">
- 현재 **YOLO 모델**이 real-time 예측 측면에서 성능이 나쁘지 않아 실무에서 가장 많이 활용되고 있음
- real-time에는 한계가 있으나 가장 성능이 좋은 모델은 **RetinaNet**

<br>

### Sliding Window 방식을 활용한 초기 object detection
<img style="padding-left: 50px" src="/image/sliding_window_example.gif" width="150px">
- object detection의 초기 기법
- window를 왼쪽 상단에서부터 오른쪽 하단으로 이동시키면서 object를 detection하는 방식
- 오브젝트가 없는 역역도 무조건 슬라이딩하며 여러 형태의 window와 scale을 스캔해야 하므로 수행시간 및 성능이 효율적이지 않음
- Region Proposal 기법의 등장 이후 활용도가 떨어졌지만 object detection의 기술적 토대 제공
<br>

### Obejct Detection의 주요 구성 요소 및 문제
**주요 구성요소**
>1. Region Proposal
2. Detection을 위한 Network 구성(feature extraction, network prediction)
3. detection을 위한 요소들(IoU, NMS, mAP, Anchor Box 등)

**주요 문제**
>1. 물체 판별(Classification) + 물체 위치 찾기(Regression)을 동시에 수행해야 함
2. 한 이미지 내에 크기, 색, 생김새가 다양한 object가 섞여 있음
3. 실시간 detection을 위해 시간 성능이 중요 
4. 명확하지 않은 이미지가 많음(노이즈 혹은 배경이 전부인 사진 등)
5. 이미지 데이터 셋의 부족

<br>

## Region Proposal(영역 추정)
- 목표 : Object가 있을 만한 후보 영역을 찾자! 
- 대표적인 기법이 Selective Search

### Selective Search
- Region Proposal의 대표적인 기법
- 컬러(color), 무늬(texture), 크기(size), 형태(shape) 등에 따라 유사한 region들을 계층적으로 그룹핑 하는 방법
<img src="/image/selective-search.png" width="600px">
<br>

**Selective Search 수행 프로세스**
>1. 초기 수 천개의 개별 Over segmentation된 모든 부분들을 bounding box로 만들어 region proposal 리스트에 추가 
2. 컬러(color), 무늬(texture), 크기(size), 형태(shape) 등에 따라 유사한 segment들을 그룹핑 
3. 위 과정을 반복하며 최종 그룹핑 된 segment들을 제안

<br>

### IoU(Intersection over Union)
모델이 예측한 bounding box와 실제 ground truth box가 얼마나 정확하게 겹치는지를 측정하는 지표
- 아래와 같은 지표로 계산 되며
<img src="/image/IoU.jpg" width="500px">
- 100%로 정확하게 겹쳐질 때의 값은 1이 됨
<img src="/image/iou-score.png" width="450px">

>IoU 값에 따라 detection 예측 성공 결정
- object detection에서 개별 object에 대한 검출 예측이 성공하였는지에 대한 여부를 IoU를 통해 결정
- 일반적으로 PASCAL VOC Challenge에서 는 IoU가 0.5이상이면 예측 성공했다고 판단
<br>

### NMS(Non Max Suppression)
object detection 시 최대한 object를 놓치지 않기 위해 많은 bounding box를 찾게 되는데, 이렇게 detected 된 수많은 bounding box 중 비슷한 위치에 있는 box를 제거하고 가장 적합한 box를 선택하는 기법 
<img src="/image/nms.png" width="700px">
<br>

**NMS 수행 프로세스**
>1. Detected 된 Bounding box별로 특정 Confidence score threshold 이하 bounding box는 먼저 제거 (ex. confidence score threshold < 0.5)
2. 가장 높은 confidence score를 가진 box 순으로 내림차순 정렬하고 아래 로직을 모든 box에 순차적으로 적용
 - 높은 confidence score를 가진 box와 겹치는 다른 box를 모두 조사하여 IoU가 특정 threshold 이상인 box를 모두 제거 (ex. IoU Threshold > 0.4)
3. 남아있는 box만 선택 

***Confidence score threshold가 높을 수록, IoU Threshold가 낮을 수록 많은 box가 제거 됨***
<br>

## Object Detection 성능 평가
### mAP(mean Average Precision)
- 실제 Object가 detected된 재현율(recall)의 변화에 따른 정밀도(precision)의 값을 평균한 성능 수치
>**정밀도와 재현율**
 - 정밀도는 모델이 positive라고 예측한 대상 중 예측 값이 실제 positive 값과 얼마나 일치하는지에 대한 비율(즉, 예측한 object가 실제 object들과 얼마나 일치하는지)
 - 재현율은 실제 positive 값 중 모델이 얼마나 실제 값을 positive라고 예측했는지에 대한 비율(즉, 실제 object를 얼마나 빠드리지 않고 잘 예측했는지)
 - Precision Recall Trade-off : 정밀도와 재현율은 상호 보완적인 관계이므로 어느 한쪽이 높아지면 다른 쪽이 낮아지게 됨
 - Precision-Recall Curve : confidence threshold의 변화에 따른 정밀도와 재현율의 변화 곡선, 이 곡선의 아랫부분 면적을 AP(Averge Precision, 평균 정밀도)라고 함
 <img src="/image/average-precision.png" width="300px">

- AP는 하나의 object에 대한 성능 수치이며, mAP는 여러 object들의 AP를 평균한 값을 의미
<img src="/image/map.png" width="500px">
<br>
