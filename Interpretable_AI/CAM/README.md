# Class Activation Map (CAM)
(** 본 내용은 레퍼런스에 언급된 자료들을 기반으로 작성되었습니다. **)

### 1. Class Activation Map (CAM) 개요:

학습한 네트워크가 이미지를 개라고 판별할 때와 고양이라고 판별할 때, 각각 이미지에서 중요하게 생각하는 영역은 다를 것이다. 이를 시각화해주는 알고리즘이 바로 Class Activation Map (CAM) 관련 알고리즘들이다.

![cam_intro](https://user-images.githubusercontent.com/7313213/138784770-95e9c32f-00d6-4419-87d8-10e47f455f02.png)

### 2. Class Activation Map 구조:

![structure](https://user-images.githubusercontent.com/7313213/138784867-92c2890e-48fd-4ced-a472-0e893c23b48a.jpg)

예측된 (Predicted) 클래스의 점수는 이전 컨볼루션 레이어와(last-convolutional layer) 다시 매핑되어 연관된 Class Activation Map을 생성해 내며, 일반적으로 Flatten을 수행하여 Fully-Connected에 연결을 수행하는 CNN과는 달리 CAM에서는 Global Average Pooling (GAP) 기법을 사용하였다.

### 3. Class Activation Map 관련 알고리즘

#### 3.1. Weakly Supervised Object Localization
![wsod](https://user-images.githubusercontent.com/7313213/138785111-f3792449-ca5e-401e-b801-2393b6ccc8d9.jpeg)

< (a) Fully-Supervised Object Detection (FSOD) uses the instance-level annotations as supervision. (b) Weakly-Supervised Object Detection (WSOD) uses the image-level annotations as supervision. >

Computer vision 분야에서 널리 알려진 Object Detection 문제는 주어진 이미지에 대해 물체가 무엇이고 어디에 있는지를 찾는게 목적이다. 이런 문제를 해결하는 모델을 학습하려면 두 가지의 annotation이 필요하다. 하나는 Object의 위치 및 크기를 표현하는 Bounding Box, 또 다른 하나는 그 물체가 무엇인지를 나타내는 물체의 category이다.

Weakly supervised object detection (WSOD)은 데이터가 좀 더 제한된 상황을 가정한다. Training data의 사진 내에 어떤 물체가 있는지 category만 주어지고, 해당 물체의 크기와 위치를 나타내는 bounding box는 빠져있다.

CAM의 경우 WSOD의 결과로 Heatmap을 얻게되고, Heatmap 근처에 Bounding box를 그리게 되면 FSOD와 유사한 결과를 획득할 수 있음이 알려져있다.


#### 3.2. Global Average Pooling (GAP)
![gap](https://user-images.githubusercontent.com/7313213/138785303-3ba2bc84-26fe-433d-b0f8-6d086a8edb25.jpg)
CNN 모델에서 Layer를 깊게 쌓으면서, model 의 parameter 수가 너무 많아지는 경향이 있어, 도대체 어디서 이 수많은 parameter 를 필요로 하는지 알아보니, FC layer (Fully-connected layer) 에서 상당량의 parameter 를 요구하고 있었다. 또한, FC layer 특성상 저 많은 parameter 를 학습할 때 필연적으로 overfitting 이 발생하며, Convolutional layer에서 FC로 연결할 때 Flatten을 수행하는데 이 때 공간에 대한 정보가 사라진다. 따라서 위와 같은 문제들을 해결하기 CAM에서는 GAP을 사용하여 Heatmap을 사용하는 알고리즘을 제시하였다.


### 4. Class Activation Map 알고리즘 동작방식

#### 4.1. 알고리즘 수식
![paper](https://user-images.githubusercontent.com/7313213/138785468-4441fa16-d7b7-43e9-b39f-95176ad42807.jpeg)

#### 4.2. CAM 생성방식 (1)
![propagation](https://user-images.githubusercontent.com/7313213/138785472-c0552d3b-e42a-47ca-a8bf-6bb3d47be191.jpeg)
![cam_from](https://user-images.githubusercontent.com/7313213/138789096-5467dc33-69b4-4d92-abec-b96b05ae8096.jpeg)

#### 4.3. CAM 생성방식 (2)
![summary](https://user-images.githubusercontent.com/7313213/138785534-19964d7b-f3c2-46d5-87ef-3187679eb388.jpeg)

각 feature map <img src="https://render.githubusercontent.com/render/math?math=f_{k}(i, j)">에 각 class에 대한 가중치 w^ck를 곱해주면 heatmap을 featuremap 개수 k 만큼 얻을수 있다. 이 heatmap 이미지를 모두 pixel-wise sum을 해주면, 하나의 heatmap을 얻을 수 있는데, 이게 바로 CAM 이다.

### 5. References

#### Class Activation Map (CAM)
- [1] [https://tyami.github.io/deep learning/CNN-visualization-Grad-CAM/](https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/)
- [2] [https://junklee.tistory.com/32](https://junklee.tistory.com/32)
- [3] [https://jinnyjinny.github.io/papers/2020/03/04/CAM/](https://jinnyjinny.github.io/papers/2020/03/04/CAM/)
- [4] [https://velog.io/@tobigs_xai/CAM-Grad-CAM-Grad-CAMpp](https://velog.io/@tobigs_xai/CAM-Grad-CAM-Grad-CAMpp)

##### Weakly Supervised Object Localization
- [1] [https://junklee.tistory.com/32](https://junklee.tistory.com/32)
- [2] [https://blog.lunit.io/2019/08/01/c-mil-continuation-multiple-instance-learning-for-weakly-supervised-object-detection/](https://blog.lunit.io/2019/08/01/c-mil-continuation-multiple-instance-learning-for-weakly-supervised-object-detection/)
- [3] Shao, Feifei, et al. "Deep Learning for Weakly-Supervised Object Detection and Object Localization: A Survey." *arXiv preprint arXiv:2105.12694* (2021)

#### Global Average Pooling (GAP)
- [1] [https://jetsonaicar.tistory.com/16](https://jetsonaicar.tistory.com/16)
- [2] [https://m.blog.naver.com/qbxlvnf11/221932118708](https://m.blog.naver.com/qbxlvnf11/221932118708)

#### CAM-Pytorch source code
- [1] [https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e8](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e8)