# Gradient-Class Activation Map (Grad-CAM)
(** 본 내용은 레퍼런스에 언급된 자료들을 기반으로 작성되었습니다. **)

### 1. Limitation of Class Activation Map (CAM):
CAM은 간단히 계산할 수 있는 유용한 툴이지만, Global Average Pooling layer를 사용해야만 한다는 한계점을 갖는다. GAP으로 대치하게되면 뒷부분을 다시 또 Fine tuning 해야하며, 마지막 Convolutional layer에 대해서만 CAM을 추출할 수 있다는 점도 한계점이 있다.

따라서, 이번 논문에서 제시된 Grad-CAM은 GAP을 쓸 필요가 없다는 점에서(Fully-connected layer 사용가능) 일반화된 CAM (Generalized CAM) 이라고도 말할 수 있다.

<br/>

### 2. Grad-CAM 알고리즘:

![formula](https://user-images.githubusercontent.com/7313213/139000003-b0a052ce-2461-4773-ad04-1a77f6d25f0c.png)

두 식의 차이점은 ReLU 함수가 추가되었다는 점과 <img src="https://render.githubusercontent.com/render/math?math=w^c_k">가 <img src="https://render.githubusercontent.com/render/math?math=a^c_k">로 변경되었다는 점이다.

<img src="https://render.githubusercontent.com/render/math?math=a^c_k">의 수식을 글로 풀어 설명해보면, k번째 feature map <img src="https://render.githubusercontent.com/render/math?math=f_{k}(i, j)">의 각 원소 i,j가 Output class c의 <img src="https://render.githubusercontent.com/render/math?math=S_c">에 주는 영향력의 평균이라고 말할 수 있다.

즉, CAM에서는 weight으로 주었던 각 feature map의 가중치를, Gradient로 대신 주었다고 생각하면 된다.

Gradient의 픽셀별 평균값인 <img src="https://render.githubusercontent.com/render/math?math=a^c_k">를 각 feature map <img src="https://render.githubusercontent.com/render/math?math=f_{k}(i, j)">에 곱해 heatmap을 만든다.

그리고 마찬가지로 pixel-wise sum을 한 후, ReLU 함수를 적용해 양의 가중치를 갖는 (중요하게 여기는) 부분을 골라내면 Grad-CAM이 된다.

<br/>

### 3. Grad-CAM 구조
![structure](https://user-images.githubusercontent.com/7313213/138999136-7d81736e-9c7e-4f0b-ba19-970c3592668a.png)

<br/>

### 4. Grad-CAM Summary
![summary](https://user-images.githubusercontent.com/7313213/138999146-ad8da7b8-4634-4c12-9ebd-a016c1e9de6d.png)

Grad-CAM은 클래스를 구분하고 이미지의 예측 영역을 국소화하는데는 뛰어나나 Guided backpropagation, Deconvolution과 같이 세밀한 부분을 강조하는 기능은 부족하다. 예를 들어, 그림의 (c)와 같이 tiger cat의 위치를 localization하여 표현할 수 있으나 모델이 왜 이 특정 사례를 'tirger cat'으로 예측하는지 명확하지 않다.

이러한 문제점을 해결하기 위하여 논문에서는 ‘Grad-CAM’에 명확한 이미지 윤곽을 리턴하는 ‘guided backpropagation’의 장점을 접목한 개념을 제시하였다 (Guided Grad-CAM). 따라서, local 한 특성을 보여주는 Grad-CAM과 Specific한 특성을 보여주는 Guided backpropagation을 pixel-wise multiplication하게되면, Local+Specific 특징을 모두 갖는 Guided Grad-CAM을 얻을 수 있다.

<br/>

### 5. References:
- [1] [http://dmqm.korea.ac.kr/activity/seminar/274](http://dmqm.korea.ac.kr/activity/seminar/274)
- [2] [https://tyami.github.io/deep learning/CNN-visualization-Grad-CAM/](https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/)
- [3] [https://jsideas.net/grad_cam/](https://jsideas.net/grad_cam/)
- [4] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." *Proceedings of the IEEE international conference on computer vision*. 2017.
