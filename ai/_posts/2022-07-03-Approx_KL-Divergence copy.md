---
layout: post
title: "[DL] KL-Divergence 근사하기"

category: study
sitemap: false
hide_last_modified: true
---

이번 포스팅은 
[Approximating KL Divergence An Unbiased Low Variance Sample Estimate](https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd) 글을 번역, 요약한 내용입니다.

Machine Learning에서는 KL-Divergence 라는 개념이 많이 활용됩니다. 그러나 다양한 이유로 정확한 값을 구하지않고 근사값을 활용하는데요, 이에대해 알아봅시다. forward와 reverse kl divergences 두 가지를 모두 다룰 예정입니다. 따라서 만약 두 개념에 익숙하지 않다면 원 저자의 [이 글](https://towardsdatascience.com/forward-and-reverse-kl-divergence-906625f1df06)을 참고하시기 바랍니다.

## 왜 근사가 필요한가
1. No analytical solution : KL divergence의 analytic solution을 구할 수 없을 수 있습니다. 예를 들어, Gaussian mixture distribution 같은 경우가 있습니다.

2. High computational complexity : 전체 KL-divergence를 구하기 위해서는 distribution space를 전부 다 더해야합니다. 근사를 시키면 이럴 필요가 없고 더 빨라지므로 유용합니다.

## 근사의 기준
KL Divergence는 'true' distribution 과 'prediction' distribution의 차이에 대한 측도입니다. 'true' distribution $p(x)$ 는 고정되어 있으며 'prediction' distribution $q(x)$ 는 우리가 컨트롤 가능하다. 우리는 $q(x)$ 로부터 sample을 뽑아 random variable화하여 근사 함수의 input으로 넣을 것이다. 만약 $p(x)$ 가 알려져 있다면, $p(x)$ 에서 sample을 뽑을 수 있지만 그런 경우가 없을 수 있기때문에 일반화를 위해 $q(x)$에서만 뽑을 것이다.

직과적으로, 근사값은 근사의 대상( original metric)과 비슷하게 동작해야한다. 우리는 이 유사성을 두 방법으로 측정할 수 있다.

1. Bias : 이상적으로는, 근사값는 불편추정량이어야한다. 즉 근사값의 평균은 original metric과 같아야한다. 

2. Variance : 0의 분산을 가진 불편추정량(deterministic)은 original metric과 정확히 일치한다. 따라서 최대한 낮은 variance를 가지는 것이 original metric과 가까운 value를 가질 확률이 높다.

## Forward KL Divergence 근사
먼저 analytic한 수식을 떠올려보자
$$KL(p\vert\vert q) = \sum\limits_xp(x)log\frac{p(x)}{q(x)} = E_p(log\frac{p(x)}{q(x)})$$

먼저 불편추정량을 만족시키기 위해:
$$E_q(approxKL) = KL(p\vert\vert q)$$
approxKL은 우리가 얻으려는 값이다. 떠올릴 수 있는 해답은 다음과 같다.
$$E_q(\frac{p(x)}{q(x)}log\frac{p(x)}{q(x)}) = \sum\limits_xq(x)\frac{p(x)}{q(x)}log\frac{p(x)}{q(x)} = \sum\limits_xp(x)log\frac{p(x)}{q(x)} = KL(p\vert\vert q)$$
따라서:
$$approxKL = \frac{p(x)}{q(x)}log\frac{p(x)}{q(x)}$$

그러나 이 추정값은 실제 KL divergence는 갖지 못하는 negative value를 가질 수 있기 때문에 variance가 아직 높다. 이를 보완하는 방법은 평균값이 0이면서 원래 근사값과 음의 상관관계를 도입하는 항을 추가하는 것이다. 다음과 같은 답을 제안한다.

$$E_q[\frac{p(x)}{q(x)} - 1] = \sum\limits_x\frac{p(x)}{q(x)}q(x) - 1 = \sum\limits_xp(x) - 1 = 0$$

$p(x)$가 유효한 확률분포로 가정되기 때문에 이렇게 표현할 수 있다. 따라서, 우리는 근사값을 다음과 같이 업데이트 할 수 있다. 

$$approxKL = \frac{p(x)}{q(x)}log\frac{p(x)}{q(x)} + \lambda(\frac{p(x)}{q(x)} - 1)$$

그리고 최소분산을 갖는 $\lambda$ 를 찾을 수 있다. 하지만 이건 데이터마다 다르고, analytic하게 풀기가 어려우므로 -1를 도입하는 것을 타협한다. 이는 어떤 경우에도 positive definite한 근사값이다. 만약 -1인 경우에 $p(x)/q(x)$ 의 plot을 그려보면 우리는 다음과 같은 그림을 얻을 수 있다. 


<p align="center">
<img src = "https://miro.medium.com/max/786/1*b-I7CN3wYT_oVbkNczguyw.png"
 width="70%" />
</p>

그러므로 forward KL divergence의 근사값의 최종 식은 다음과 같다.

$$approxKL = rlogr-(r-1)$$
$$r = \frac{p(x)}{q(x)}$$

## Reverse KL Divergence 근사
Forward KL divergence 를 근사한 방식으로 Reverse KL divergence도 근사할 수 있다.

일단 analytical 식을 떠올려보자
$$KL(q\vert\vert p) = \sum\limits_xq(x)log\frac{q(x)}{p(x)} = E_p(log\frac{q(x)}{p(x)})$$

다시 한번, 간단한 불편추정량을 얻어보자. 해답은 다음과 같다.(우리는 q(x)는 sample할 수 있음을 생각하자)

$$E_q(log\frac{q(x)}{p(x)}) = \sum\limits_xq(x)log\frac{q(x)}{p
(x)} = KL(q\vert\vert p)$$

따라서 추정량은 

$$approxKL = log\frac{q(x)}{p(x)}$$

이는 forward KL divergence의 불편추정량과 정확히 동일한 문제를 가진다. 음수를 포함하기 때문에, 분산이 커진다는 점이다. 따라서 같은 항을 추가한다. 


$$approxKL = log[\frac{q(x)}{p(x)}] + \lambda(\frac{p(x)}{q(x)} -1) = \lambda(\frac{p(x)}{q(x)} -1) - log\frac{p(x)}{q(x)}$$

이제 동일하게 최소 분산을 가지는 $\lambda$를 찾으면 되지만 이는 상황마다 다르므로 1을 도입하는 것으로 타협한다. $p(x)/q(x)$ plot을 그려보면 다음과 같다.

<p align="center">
<img src = "https://miro.medium.com/max/770/1*4UPk-eVv8_YFdvfCON-Qog.png"
 width="70%" />
</p>

그러므로 최종 reverse KL divergence 의 근사값은 다음과 같다.

$$approxKL = (r-1) - logr$$
$$r = \frac{p(x)}{q(x)}$$


## 추가내용

<!-- 
## 추가
OpenAI baseline, Stable baseline, TF agent 를 살펴봤으나 
TF-agent 는 ^2 stable baseline은 이거, openai baseline은 ?? -->

상용 오픈소스는 KL divergence를 어떻게 사용하고 있는지 확인하였다. 원글의 소스가 된 포스팅의 저자는 위 내용을 바탕으로 stable baseline 에 contribution을 넣었으므로 stable baseline에는 반영되어 있다. 다른 오픈소스는 어떨까?

- OpenAI baseline
- TF-agent
    - 근사없이 unbiased reverse KL divergence를 사용한다. 