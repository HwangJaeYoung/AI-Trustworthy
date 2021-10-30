# Local Interpretable Model-agnostic Explanations (LIME) 
(** 본 내용은 레퍼런스에 언급된 자료들을 기반으로 작성되었습니다. **)

### 1. LIME 개요:
**LIME (locally interpretable model-agnostic explanations)**: 모델의 개별 예측값을 설명하기 위한 알고리즘

복잡한 모형을 해석이 가능한 심플한 모형으로 locally approximation을 수행하여 설명을 시도한다. 이름에서 알 수 있듯 전체 모델이 아닌 개별 prediction의 근방에서만 해석을 시도한다는 점과 어떠한 모델 (딥러닝, 랜덤 포레스트, SVM 등) 및 데이터 형식도(이미지, 텍스트, 수치형) 적용이 가능하다는 특징이 있다.

- 오늘날 신경망과 같이 복잡성이 높은 머신러닝 모델을 사용하는 일반적인 상황에서, 예측 결과에 대하여 전역적으로 완벽한 설명을 제시하는 것은 현실적으로 매우 어려운 일이다. 비록 머신러닝 모델의 전체 예측 결과에 대하여 완벽한 설명을 한 번에 제시하는 것은 불가능하더라도, 적어도 사용자가 관심을 가지는 몇 개의 예측 결과에 한하여 즉각적으로 설명을 제시해 줄 수 있다.

![lime_intro](https://user-images.githubusercontent.com/7313213/139409142-8f4c9732-9ae5-4f06-88ff-c5fbbe85449b.png)

<br/>


####  LIME을 활용한 Inception 모델의 예측 결과

![frog1](https://user-images.githubusercontent.com/7313213/139522582-aae32a13-1f17-46be-8e02-a805e83ac275.png)

LIME은 Inception 모델에 주어진 이미지를 개구리라고 예측한 데에는 개구리의 얼굴이 가장 중요한 역할을 했다고 설명.

그런데 당구대와 풍선은 왜 나온 것일까?

⇒ 초록색 배경에 당구공 같이 생긴 것들이 있으니 모델이 당구대라고 생각

⇒ 개구리가 들고 있는 하트는 빨간색 풍선을 닮아 모델이 풍선이라고 예측

이렇듯 LIME을 통해 왜 모델이 이런 예측들을 했는지 이해할 수 있음

<br/>

***LIME 핵심: 입력값을 조금 바꿨을 때 모델의 예측값이 크게 바뀌면, 그 변수는 중요한 변수이다.***

1. 먼저 이미지를 superpixel이라고 불리는 해석 가능한 요소로 쪼개는 전처리 과정 수행

![frog2](https://user-images.githubusercontent.com/7313213/139522583-a340832e-339a-4dda-ab94-7b3cb0cb35cf.png)

<br/>

2. 그리고 나서 변수에 약간의 변화(perturbation)를 준다. 이미지의 경우에는 superpixel 몇개를 찝어서 회색으로 가리고 모델을 통해 예측값을 구한다.

   만약 예측값이 많이 변하면 가렸던 부분이 중요했다는 것을 알 수 있으며, 반대로 예측값이 많이 달라지지 않았으면 가렸던 부분이 별로 중요하지 않다라는 것을 알 수 있다.

![frog3](https://user-images.githubusercontent.com/7313213/139522584-1d72d149-e0bb-4c38-81b0-7eec0f0a0ba8.png)

<br/>

3. 예시에서 원본 사진은 0.54 확률로 개구리로 예측이 된다.
    
    이 사진을 첫번째 변형(perturbed instance)처럼 가렸더니 개구리일 확률이 0.85로 높아졌다. 사진에서 남은 부분이 개구리라고 예측하는 데 중요한 요소라는 것을 알 수 있다.
    
    두번째 변형처럼 가렸더니 개구리일 확률이 0.00001로 매우 낮아졌다. 그러면 방금 가린 부분이 개구리라고 판단하는 데 중요한 요소였다는 것을 알 수 있음.
    
    세번째 변형처럼 가리면 개구리일 확률이 별로 변하지 않는다. 이때 가린 부분은 개구리라고 판단하는 데 별로 중요하지 않았다는 것을 알 수 있다. 
    
    ⇒ 이렇게 여러번의 과정을 거친 뒤 결국 어떤 superpixel이 개구리라고 판단하는 데 가장 중요했는지 찾는 것이 LIME의 핵심.

<br/>

### 2. 이미지 판별에 대한 LIME 활용:

#### 2.1 슈퍼픽셀 (Super-pixel)
![super_pixel](https://user-images.githubusercontent.com/7313213/139409482-9b6eb8e9-e0f8-41e4-bc57-c9cd73b4787a.png)

- LIME에서는 이미지를 여러 "세그먼트" 또는 "슈퍼 픽셀"로 분할하고 임의의 슈퍼 픽셀을 켜고 끄는 방식으로 섭동 이미지를 획득한다.

- 이미지를 슈퍼픽셀로 분할하는 이유는 상관관계가 있는 픽셀을 함께 그룹화하여 최종 예측에 미치는 영향을 살펴보려고 하기 때문이다.

#### 2.2 LIME 적용

(1) Generate random perturbatnios for input image

   이미지의 경우 LIME은 이미지의 슈퍼픽셀 일부를 켜고 끄는 방식으로 이미지 섭동 (Perturbations)을 생성한다.
   ![image_segment](https://user-images.githubusercontent.com/7313213/139409490-16f322f0-a521-41b1-84ed-f248da971198.png)    

<br/>

(2) Predict the class of each of the perturbed images

   섭동 된 이미지를 사용하여 예측 결과를 얻는다.

   Inception v3 모델을 사용하는 경우 1000개의 클래스가 예측되며, 나중에 Surrogate Model을 학습하는 할 때 래브라도 클래스만 사용한다.

   여기서 얻어진 래브라도 확률을 예측값으로 (섭동이미지, 래브라도 예측값) Surrogate Model을 학습시킬 데이터 셋을 새로 구축한다.

   ![predict](https://user-images.githubusercontent.com/7313213/139409493-db66cb04-e2fd-4146-a987-4e80e2e1da9c.png)

<br/>

(3) Compute weight (importance) for the perturbations

   거리 값을 사용하여 각 섭동이 원본 이미지로부터 얼마나 떨어져 있는지 평가한다.

   원본 이미지에서 가까이 있을수록 가중치를 크게 주고 멀리 있으면 작게 만들어 원본 이미지  주변 Local의 특성을 최대한 살린다

<br/>

(4) Fit a explainable liner model using the perturbations, predictions and weight

   이전 단계에서 얻은 정보를 사용하여 선형 모델을 학습시킨다.

   학습 된 모델에서 각 계수 (coefficient)의 Rank를 부여하였을 때 가장 높은 값 순으로 래브라도 클래스의 예측에 가장 많은 영향을 미친 슈퍼픽셀이 된다.

   ![lime_cal](https://user-images.githubusercontent.com/7313213/139410320-b0dfb47a-fb69-4e1e-9843-143aa728f8a7.png)


<br/>

### 3. LIME Summary

![lime_summary](https://user-images.githubusercontent.com/7313213/139408670-23d2de61-2611-4d85-a29e-a798fa788d43.png)

- 래브라도의 예측에 가장 많이 영향을 준 이미지 영역(슈퍼 픽셀)이 활성화 된다.
- 인공지능 모델이 래브라도 클래스를 어떤 근거로 예측하고 있는지 확인할 수 있다.
- 즉, LIME이 특정 예측을 반환하는 이유를 이해함으로써 인공지능 모델에 대한 신뢰도를 높일 수 있음.

<br/>

### 4. References

- [1] [https://dreamgonfly.github.io/blog/lime/](https://dreamgonfly.github.io/blog/lime/)
- [2] [https://nhlmary3.tistory.com/entry/LIME-Locallly-Interpretable-Modelagnostic-Explanation](https://nhlmary3.tistory.com/entry/LIME-Locallly-Interpretable-Modelagnostic-Explanation)
- [3] [https://realblack0.github.io/2020/04/27/explainable-ai.html](https://realblack0.github.io/2020/04/27/explainable-ai.html)
- [4] [https://sualab.github.io/introduction/2019/08/30/interpretable-machine-learning-overview-1.html](https://sualab.github.io/introduction/2019/08/30/interpretable-machine-learning-overview-1.html)
- [5] [http://dmqm.korea.ac.kr/activity/seminar/297](http://dmqm.korea.ac.kr/activity/seminar/297)
- [6] [https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13](https://towardsdatascience.com/interpretable-machine-learning-for-image-classification-with-lime-ea947e82ca13)
- [7] Interpretable AI, Ajay Thampi, Manning Publications