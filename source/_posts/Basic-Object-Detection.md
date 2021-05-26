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

#### Python을 통한 Selective Search 구현
- <code>pip install selectivesearch</code> 를 통해 라이브러리 설치 

<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div><div style="line-height:130%">7</div><div style="line-height:130%">8</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">import</span>&nbsp;selectivesearch</div><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">import</span>&nbsp;cv2</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">img&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;cv2.imread(<span style="color:#ffd500">'./image/test.jpg'</span>)&nbsp;<span style="color:#999999">#&nbsp;이미지&nbsp;로드&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">img_rgb&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;cv2.cvtColor(img,&nbsp;cv2.COLOR_BGR2RGB)</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.figure(figsize<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>(<span style="color:#c10aff">8</span>,&nbsp;<span style="color:#c10aff">8</span>))</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.imshow(img_rgb)</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.show()</div></div><div style="text-align:right;margin-top:-13px;margin-right:5px;font-size:9px;font-style:italic"><a href="http://colorscripter.com/info#e" target="_blank" style="color:#4f4f4ftext-decoration:none">Colored by Color Scripter</a></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>

<img src="/image/audrey_original.png" width="250px"/>

<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#999999">#selectivesearch.selective_search()는&nbsp;이미지의&nbsp;Region&nbsp;Proposal정보를&nbsp;반환&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">_,&nbsp;regions&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;selectivesearch.selective_search(img_rgb,&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scale<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">100</span>,&nbsp;<span style="color:#999999">#&nbsp;bounding&nbsp;box&nbsp;scale&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;min_size<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">2000</span>)&nbsp;<span style="color:#999999">#&nbsp;rect의&nbsp;최소&nbsp;사이즈</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">regions[:<span style="color:#c10aff">5</span>]</div></div><div style="text-align:right;margin-top:-13px;margin-right:5px;font-size:9px;font-style:italic"><a href="http://colorscripter.com/info#e" target="_blank" style="color:#4f4f4ftext-decoration:none">Colored by Color Scripter</a></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>

```
[{'rect': (0, 0, 58, 257), 'size': 7918, 'labels': [0.0]},
 {'rect': (16, 0, 270, 50), 'size': 5110, 'labels': [1.0]},
 {'rect': (284, 0, 90, 420), 'size': 6986, 'labels': [2.0]},
 {'rect': (59, 14, 262, 407), 'size': 3986, 'labels': [3.0]},
 {'rect': (62, 17, 256, 401), 'size': 5282, 'labels': [4.0]}]
```
반환된 regions 변수는 리스트 타입으로 세부 원소로 딕셔너리를 가지고 있음. 
* rect 키값은 x,y 시작 좌표와 너비, 높이 값을 가지며 이 값이 Detected Object 후보를 나타내는 Bounding box임. 
* size는 Bounding box의 크기
* labels는 해당 rect로 지정된 Bounding Box내에 있는 오브젝트들의 고유 ID
* 아래로 내려갈 수록 특성이 비슷한 것들이 합쳐지고, 너비와 높이 값이 큰 Bounding box이며 하나의 Bounding box에 여러개의 오브젝트가 있을 확률이 커짐. 

<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div><div style="line-height:130%">7</div><div style="line-height:130%">8</div><div style="line-height:130%">9</div><div style="line-height:130%">10</div><div style="line-height:130%">11</div><div style="line-height:130%">12</div><div style="line-height:130%">13</div><div style="line-height:130%">14</div><div style="line-height:130%">15</div><div style="line-height:130%">16</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#999999">#&nbsp;Bounding&nbsp;Box&nbsp;시각화&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">green_rgb&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;(<span style="color:#c10aff">125</span>,&nbsp;<span style="color:#c10aff">255</span>,&nbsp;<span style="color:#c10aff">51</span>)</div><div style="padding:0 6px; white-space:pre; line-height:130%">img_rgb_copy&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;img_rgb.copy()</div><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">for</span>&nbsp;rect&nbsp;<span style="color:#ff3399">in</span>&nbsp;cand_rects:</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;left&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;rect[<span style="color:#c10aff">0</span>]</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;top&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;rect[<span style="color:#c10aff">1</span>]</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#999999">#&nbsp;rect[2],&nbsp;rect[3]은&nbsp;너비와&nbsp;높이이므로&nbsp;우하단&nbsp;좌표를&nbsp;구하기&nbsp;위해&nbsp;좌상단&nbsp;좌표에&nbsp;각각을&nbsp;더함.&nbsp;</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;right&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;left&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">+</span>&nbsp;rect[<span style="color:#c10aff">2</span>]</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;bottom&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;top&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">+</span>&nbsp;rect[<span style="color:#c10aff">3</span>]</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;img_rgb_copy&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;cv2.rectangle(img_rgb_copy,&nbsp;(left,&nbsp;top),&nbsp;(right,&nbsp;bottom),&nbsp;color<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>green_rgb,&nbsp;thickness<span style="color:#0086b3"></span><span style="color:#ff3399">=</span><span style="color:#c10aff">2</span>)</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.figure(figsize<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>(<span style="color:#c10aff">8</span>,&nbsp;<span style="color:#c10aff">8</span>))</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.imshow(img_rgb_copy)</div><div style="padding:0 6px; white-space:pre; line-height:130%">plt.show()</div></div><div style="text-align:right;margin-top:-13px;margin-right:5px;font-size:9px;font-style:italic"><a href="http://colorscripter.com/info#e" target="_blank" style="color:#4f4f4ftext-decoration:none">Colored by Color Scripter</a></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>

<img src="/image/audrey-bounding.png" width="250px"/>
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

#### Python을 통한 IoU 계산
<div class="colorscripter-code" style="color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important; position:relative !important;overflow:auto"><table class="colorscripter-code-table" style="margin:0;padding:0;border:none;background-color:#272727;border-radius:4px;" cellspacing="0" cellpadding="0"><tr><td style="padding:6px;border-right:2px solid #4f4f4f"><div style="margin:0;padding:0;word-break:normal;text-align:right;color:#aaa;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="line-height:130%">1</div><div style="line-height:130%">2</div><div style="line-height:130%">3</div><div style="line-height:130%">4</div><div style="line-height:130%">5</div><div style="line-height:130%">6</div><div style="line-height:130%">7</div><div style="line-height:130%">8</div><div style="line-height:130%">9</div><div style="line-height:130%">10</div><div style="line-height:130%">11</div><div style="line-height:130%">12</div><div style="line-height:130%">13</div><div style="line-height:130%">14</div><div style="line-height:130%">15</div></div></td><td style="padding:6px 0;text-align:left"><div style="margin:0;padding:0;color:#f0f0f0;font-family:Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;line-height:130%"><div style="padding:0 6px; white-space:pre; line-height:130%"><span style="color:#ff3399">def</span>&nbsp;compute_iou(cand_box,&nbsp;gt_box):</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#999999">#&nbsp;Calculate&nbsp;intersection&nbsp;areas</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;x1&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;np.maximum(cand_box[<span style="color:#c10aff">0</span>],&nbsp;gt_box[<span style="color:#c10aff">0</span>])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;y1&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;np.maximum(cand_box[<span style="color:#c10aff">1</span>],&nbsp;gt_box[<span style="color:#c10aff">1</span>])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;x2&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;np.minimum(cand_box[<span style="color:#c10aff">2</span>],&nbsp;gt_box[<span style="color:#c10aff">2</span>])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;y2&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;np.minimum(cand_box[<span style="color:#c10aff">3</span>],&nbsp;gt_box[<span style="color:#c10aff">3</span>])</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;intersection&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;np.maximum(x2&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;x1,&nbsp;<span style="color:#c10aff">0</span>)&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">*</span>&nbsp;np.maximum(y2&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;y1,&nbsp;<span style="color:#c10aff">0</span>)&nbsp;<span style="color:#999999">#&nbsp;width&nbsp;*&nbsp;height&nbsp;(x2에서&nbsp;x1을&nbsp;뺀&nbsp;값이&nbsp;width,&nbsp;y2에서&nbsp;y1을&nbsp;뺀&nbsp;값이&nbsp;height&nbsp;이므로)</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;cand_box_area&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;(cand_box[<span style="color:#c10aff">2</span>]&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;cand_box[<span style="color:#c10aff">0</span>])&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">*</span>&nbsp;(cand_box[<span style="color:#c10aff">3</span>]&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;cand_box[<span style="color:#c10aff">1</span>])&nbsp;<span style="color:#999999">#&nbsp;width&nbsp;*&nbsp;height</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;gt_box_area&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;(gt_box[<span style="color:#c10aff">2</span>]&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;gt_box[<span style="color:#c10aff">0</span>])&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">*</span>&nbsp;(gt_box[<span style="color:#c10aff">3</span>]&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;gt_box[<span style="color:#c10aff">1</span>])&nbsp;<span style="color:#999999">#&nbsp;width&nbsp;*&nbsp;height</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;union&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;cand_box_area&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">+</span>&nbsp;gt_box_area&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">-</span>&nbsp;intersection&nbsp;<span style="color:#999999">#&nbsp;실제box와&nbsp;예측box의&nbsp;합에서&nbsp;intersection을&nbsp;뺌</span></div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;iou&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">=</span>&nbsp;intersection&nbsp;<span style="color:#0086b3"></span><span style="color:#ff3399">/</span>&nbsp;union</div><div style="padding:0 6px; white-space:pre; line-height:130%">&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#ff3399">return</span>&nbsp;iou</div></div><div style="text-align:right;margin-top:-13px;margin-right:5px;font-size:9px;font-style:italic"><a href="http://colorscripter.com/info#e" target="_blank" style="color:#4f4f4ftext-decoration:none">Colored by Color Scripter</a></div></td><td style="vertical-align:bottom;padding:0 2px 4px 0"><a href="http://colorscripter.com/info#e" target="_blank" style="text-decoration:none;color:white"><span style="font-size:9px;word-break:normal;background-color:#4f4f4f;color:white;border-radius:10px;padding:1px">cs</span></a></td></tr></table></div>

- 실제 bounding box와 후보 bounding box가 있을 때, 둘 중에서  x1과 y1좌표는 max값, x2와 y2좌표는 min값을 선택하게 되면 그 좌표가 Intersection area가 되며
- 두 개의 box를 더한 후 intersection을 빼준 값이 Union area 
- 마지막으로 intersection을 union으로 나누어 주면 IoU값을 얻을 수 있음 
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

### Image Resolution / FPS / Detection 성능 상관 관계
<img src="/image/resolution-detection-score.png" width="400px">
일반적으로 이미지 해상도(Image Resolution)가 높을 수록 Detection성능이 좋아지지만 이미지를 처리하는 시간(FPS)이 오래걸림
- High Resolution -> High Detection Score -> Low FPS
- Low Resolution -> Low Detection Score -> High FPS
<br>

## Object Detection을 위한 주요 데이터 셋
- [Pascal-VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) - XML format, 20개의 오브젝트 카테고리 
- [MS-COCO](http://cocodataset.org/#home) - json format, 80개의 오브젝트 카테고리
- [Google Open Images](https://opensource.google/projects/open-images-dataset) - csv format, 600개의 오브젝트 카테고리
