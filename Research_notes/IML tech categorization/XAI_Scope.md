# Interpretable Machine Learning tech categorization

(*** 본 내용은 레퍼런스에 언급된 블로그를 참조하여 작성되었습니다. ***)

Interpretable Machine Learning (IML) 기술은 크게 3가지 관점에서 분류 할 수 있다. (Complexity, Scope, Dependency)

|관점|분류|
|---|---|
|Complexity|Intrinsic vs. Post-hoc|
|Scope|Global vs. Local|
|Dependency|Model-specific vs. Model-agnostic|

## 1. Complexity

모델의 복잡성 (Complexity)은 해석력과 깊은 연관이 있다. 모델이 복잡할수록 사람이 해석하기가 더 어려우며, 반대로 모델이 단순할수록 사람이 해석하기는 더 용이하다. 그러나 어려운 문제를 해결하기 위해서는 복잡한 구조가 (ex. DNN) 유리하기 때문에 모델의 복잡성과 해석력은 서로 Trade-off 관계가 있다.

### 1.1 Intrinsic

내재적으로 해석력을 확보하고 있는 머신러닝 모델을 ‘intrinsic(본래 갖추어진)’하다고 지칭한다. 예를 들어 의사 결정 나무(Desicion Tree)의 경우 그 자체적으로 해석력을 이미 확보하고 있다고 볼 수 있으며, 이를 두고 ‘투명성(transparency)‘을 확보하고 있다고도 한다. 

![tree](https://user-images.githubusercontent.com/7313213/138577190-9ae446f2-a8b8-4275-b02e-9de37b08b579.png)

### 1.2 Post-hoc

복잡성이 극도로 높은 전형적인 머신러닝 모델로는 신경망(neural network), 즉 딥러닝 모델이 있다. 딥러닝은 내부적으로 복잡한 연결 관계를 지니고 있으며, 이로 인해 예측 결과를 도출하는 과정에서 하나의 입력 성분값 (이미지 데이터의 경우 픽셀 1개의 값)이 어떻게 기여했는지 의미적으로 해석하는 것이 대단히 어렵다. 따라서, 모델 자체가 해석력을 지니지 않을 경우, 모델의 예측 결과는 사후(post-hoc)에 해석할 수 밖에 없으며 기계학습 및 딥러닝 분야에서 해석 가능한 기법이 대부분 Post-hoc에 속한다.

이렇게 복잡성이 높은 머신러닝 모델에 대하여 적용할 수 있는 대안으로, 비교적 간단한 형태를 지니는 별도의 ‘설명용 모델 (interpretable model)’을 셋팅하고(e.g. 의사 결정 나무, 희소 선형 모델, 규칙 리스트 등) 이를 설명 대상이 되는 머신러닝 모델에 갖다 붙여 적용하는 방법을 시도할 수 있다.

![surrogate](https://user-images.githubusercontent.com/7313213/138577224-ffbb3364-b606-4487-a998-580951be0a59.jpg)

## 2. Scope

IML 방법론이 머신러닝 모델의 예측 결과들에 대하여 ‘전역적으로(globally)’ 완벽하게 설명을 수행할 수 있는 경우, 이를 ‘**global**(전역적)’ 방법이라한다. 반면 설명 대상 머신러닝 모델의 복잡성이 증가할수록 모든 예측 결과에 대하여 설명을 제시하는 것이 어려우며 이 때문에 몇몇 IML 방법론들은 완벽하게 ‘전역적인’ 설명을 포기하는 대신, 모델의 어느 예측 결과에 대하여 적어도 그와 유사한 양상을 나타내는 ‘주변’ 예측 결과들에 한해서는 ‘국소적으로(locally)’ 설명을 제시할 수 있도록 디자인 되었다. 이를 ‘**local**(국소적)’ 방법이라고 지칭한다.

2.1 Global

- Global 기법은 모델의 로직과 관련된 이해를 바탕으로, 모델이 예측하는 모든 결과를 설명한다.
- Intrinsic 모델은 모델의 구조로부터 모든 예측 결과에 대한 설명이 가능하므로 태생적으로 Global 기법에 속한다. (Decision Tree, Falling Rule List 등)
- Global은 이상적인 설명 기법이지만 Post-hoc을 Global로 구현하기는 현실적으로 까다롭다. 설명 측면에서, 모든 예측에 대해서는 일정한 설명력을 갖출 수 있더라도, 개별 예측 결과의 특징을 설명하는 능력은 다소 떨어질 수 있다.

2.2 Local

- Local 기법은 특정한 하나의 예측 결과만 설명한다. Global 기법에 대비해서 Local 기법은 설명할 범위가 적어서 비교적 실현성 있고 비용이 적게 든다. 또한, 전반적인 예측 성향은 설명하지 못하더라도 하나 또는 소수의 예측 결과는 완벽에 가깝게 설명할 수 있다.
- 현실적으로 매번 예측할 때마다 설명을 요구하지는 않는다. 이슈가 발생한 예측만 설명하는 것이 현실적이다. 설명이 필요할 때, 적어도 해당 이슈에 대해서는 잘 설명할 수 있다는 점에서 Global 기법보다 실용적이다.
- 오늘날 신경망과 같이 복잡성이 높은 머신러닝 모델을 사용하는 일반적인 상황에서, 예측 결과에 대하여 전역적으로 완벽한 설명을 제시하는 것은 현실적으로 매우 어려운 일이다. 비록 머신러닝 모델의 전체 예측 결과에 대하여 완벽한 설명을 한 번에 제시하는 것은 불가능하더라도, 적어도 사용자가 관심을 가지는 몇 개의 예측 결과에 한하여 즉각적으로 설명을 제시해 줄 수 있다.

![local](https://user-images.githubusercontent.com/7313213/138577416-88b2f109-d482-44f2-8481-7467dcc326dd.jpg)
## 3. Dependency

해석 방법론이 어느 특정한 종류의 머신러닝 모델에 특화되어 작동하는지, 혹은 모든 종류의 머신러닝 모델에 범용적으로 작동하는지에 따라 분류할 수도 있다.

### 3.1 Model-specific

특정 종류의 모델만 적용할 수 있는 설명 기법을 Model-specific(모델 특정적)이라고 하며,  Intrinsic 기법은 모델 자체가 가지고 있는 특성을 이용하므로 타 모델에서 적용할 수 없는 전형적인 Model-specific이다.  CNN 계열에서만 쓸 수 있는 시각화 해석 기법은 모두 Model-specific에 해당한다.

### 3.2 Model-agnostic

신경망과 같이 복잡성이 높아 자체적인 해석력을 확보하기 어려운 머신러닝 모델의 경우 내부의 구조는 사람이 알 수 없다.  모델을 설명하기 위해서는 모델 밖에서 근거를 찾아야 한다는 Model-agnostic(모델 불가지론)은 모델의 어떠한 특성도 이용하지 않으며, 모델에 상관없이 적용 가능한 특징이 있다. 따라서 해석력 확보를 위하여 별도의 post-hoc 방법을 통해 설명용 모델을(Surrogate) 생성하고 이를 갖다 붙여 활용한다.

** intrinsic과 model-specific, post-hoc과 model-agnostic은 서로 간의 관점에 차이가 있을 뿐, 실질적으로는 동시적으로 적용될 수 있는 특성이라고 봐도 크게 무리가 없다고 할 수 있다.

## 4. Summary

![summary](https://user-images.githubusercontent.com/7313213/138577287-7adfdb0a-4d1e-46bb-b83c-56623ed84fc6.png)
## 5. References:
- [1] [https://realblack0.github.io/2020/04/27/explainable-ai.html](https://realblack0.github.io/2020/04/27/explainable-ai.html)

- [2] [https://sualab.github.io/introduction/2019/08/30/interpretable-machine-learning-overview-1.html](https://sualab.github.io/introduction/2019/08/30/interpretable-machine-learning-overview-1.html)