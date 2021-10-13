# Image-specific Saliency

### 1. Image-specific Saliency 개요:

컨볼루션 신경망의 Attribution을 보여주기 위한 대표적인 수단이 ‘Saliency Map(현저성 맵)’이다. 보통 Saliency Map은 이미지 상의 두드러진 부분을 지칭하나, 컨볼루션 신경망의 예측 결과에 대한 설명의 맥락에서는, 예측 결과를 이끌어낸 이미지 상의 주요한 부분을 표현하기 위한 목적으로 생성된다.
컨볼루션 신경망의 예측 결과로부터 Saliency Map을 도출하기 위한 가장 간단한 방법은, 예측 클래스의 입력 이미지 X에 대한 gradient ∂yc/∂X를 계산하는 것이다. 마치 앞서 소개했던 Maximization by Optimization과 유사해 보일 것인데, Maximization by Optimization이 랜덤한 이미지에서 출발하여 feature map의 gradient를 반복적으로 더해주는 gradient ascent를 통해 가상의 이미지를 생성하였다면, Saliency Map의 경우 실제 입력 이미지에 대한 예측 클래스의 gradient를 한 번만 계산하여 이를 그대로 활용한다는 점이 차이라고 할 수 있다.

![saliency-map-with-gradient-concept](https://user-images.githubusercontent.com/7313213/137118819-92f9a7f5-9612-4554-a30a-50e9ef8c9800.jpg)

### 2. Image-specific Saliency 처리순서:

Gradient ascent를 이용하여 클래스 이미지를 생성했던 방법과 비슷한데, **random image가 아니라 특정 image를 입력해 해당 image를 classification하는 데에 큰 영향을 끼친 부분을 heatmap으로 표시**한다.

과정은 다음과 같다.

1. 타겟 입력 이미지를 CNN에 넣어 class score를 얻는다.
2. Backpropagation으로 입력 이미지의 gradient를 구한다.
3. 얻어진 gradient에 절댓값을 취하거나, 제곱을 하여 절대적인 크기(magnitude)를 구한다.
    - 어느 쪽으로 바뀌는지보다 해당 영역이 얼마나 큰 영향을 끼치는가가 중요하므로, 부호를 버리고 magnitude를 측정한다.
4. 해당 gradient magnitude map을 시각화한다. 필요에 따라 1-3을 반복하여 accumulate할 수도 있다.

Class visualization의 gradient ascent와 다른 것은, 아무 이미지나 넣어 해당 클래스에 대한 모델의 예상치 추측을 하는것이 아니라, **특정 이미지에 대한 모델의 판단 요인을 찾는다는 것이다.** 즉, 현재 데이터가 어떻게 해석되는지를 보고싶은 것이므로 data-dependent하다.

### 3. Image-specific Saliency 알고리즘 정리

- Linear score model

![linear score function](https://user-images.githubusercontent.com/7313213/137119546-eddb0879-d1a8-460a-85fc-128e32458d20.JPG)

Pixel들을 rank시키기 위해서 간단한 Linear socre model 도출하였다. 즉, image의 각 pixel들이 CNN 모델의 결정에 미치는 중요도를 어떻게 측정해볼까라고 생각하여 선형 모델을 도입하였음. Linear score model을  사용하여 (Conceptual) 이미지의 각 Pixel들이 class C라는 결정을 내리는데 있어 어느정도의 영향력을 미쳤는지 판단할 수 있다고 하자, 이 때 class score에 가장 영향을 많이 미치는 항목은 W (Magnitude of the weights)임을 알 수 있다 (여기서 weight(WcT)는 ConvNet의 backpropagation을 통한 weight가아니다.). 이를 기반으로 Linear score model을 CNN 모델에 적용할 수 있을까라고 하면 할 수 없다. 그 이유는 CNN 모델은 non-linear model 이기 때문이다. 따라서, 테일러 급수를 활용한 Linear approximation (선형근사)를 통해서 CNN의 class score를 구할 수 있다.

Given an image I_zero, a class c, and a classification ConvNet with the class score function Sc(I), we would like to rank the pixels of I_zero based on their influence on the score Sc(I_zero).

⇒ 해당 문장에서 I는 I0, I1, I2, .., In이 될 수 있음.

- 테일러 급수

![taylor_series](https://user-images.githubusercontent.com/7313213/137119549-c157e277-222e-49ee-9b75-a04cf933112c.jpg)

- 테일러 급수를 활용한 선형근사

![score_function_for_cnn](https://user-images.githubusercontent.com/7313213/137119551-d5b631a9-bd26-4083-a31c-1ef17846a8b2.JPG)

-위 도출된식에서의 특징은 w만 Image-specific saliency를 도출 하는데 사용된다.

-소스코드에서 max(score)를 활용하여 class score를 구하는 이유는 softmax와 같은 계층에 들어가기 전의 계층(penultimate)에서 도출된 결과가 가장 높은 값이 결국 그 클래스의 예측을 나타나는 값이 되기 때문이다. (Class scores, by the way, are the values in the output layer that the neural network assigns to classes before the softmax, so they’re not probabilities, but they’re directly related to probabilities through a function like softmax.)

-매클로린급수를 사용하여서 식을 간단히 했다고 보았을 때 I_zero = 0 vertor image이다. 따라서, I_zero를 기준으로 Linear approximation을 수행한다면 향후 정규화된 이미지를 삽입하여 각 이미지의 class score를 구할 수 있음.

![paper_review](https://user-images.githubusercontent.com/7313213/137119134-374f57b2-b21d-4e1e-9b36-d750fdf40886.jpg)

### 4. References
- [https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4](https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4)
- [https://simonjisu.github.io/paper/2020/03/12/deepinsidecnn.html](https://simonjisu.github.io/paper/2020/03/12/deepinsidecnn.html)
- [https://blogik.netlify.app/BoostCamp/U_stage/41_cnn_visualization/](https://blogik.netlify.app/BoostCamp/U_stage/41_cnn_visualization/)
- [https://glassboxmedicine.com/2019/06/21/cnn-heat-maps-saliency-backpropagation/](https://glassboxmedicine.com/2019/06/21/cnn-heat-maps-saliency-backpropagation/)
- [https://www.cognex.com/ko-kr/blogs/deep-learning/research/overview-interpretable-machine-learning-2-interpreting-deep-learning-models-image-recognition](https://www.cognex.com/ko-kr/blogs/deep-learning/research/overview-interpretable-machine-learning-2-interpreting-deep-learning-models-image-recognition)
