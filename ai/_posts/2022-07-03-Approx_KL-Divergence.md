---
layout: post
title: "[DL] "

category: study
sitemap: false
hide_last_modified: true
---

[Approximating KL Divergence
An Unbiased Low Variance Sample Estimate](https://towardsdatascience.com/approximating-kl-divergence-4151c8c85ddd) 글 번역한 것입니다.

KL divergence 값을 표본으로 근사하는 방법을 알아보겠습니다. forward와 reverse kl divergences 두 가지를 모두 다룰 예정입니다. 따라서 만약 두 개념에 익숙하지 않다면 원 저자의 [이 글](https://towardsdatascience.com/forward-and-reverse-kl-divergence-906625f1df06)을 참고하시기 바랍니다.

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
approxKL은 우리가 얻으려는 값이다. 해답은 다음과 같다.
$$E_q(\frac{p(x)}{q(x)}log\frac{p(x)}{q(x)}) = \sum\limits_xq(x)\frac{p(x)}{q(x)}log\frac{p(x)}{q(x)} = \sum\limits_xp(x)log\frac{p(x)}{q(x)} = KL(p\vert\vert q)$$
따라서:
$$approxKL = \frac{p(x)}{q(x)}log\frac{p(x)}{q(x)}$$

그러나 이 추정값은 실제 KL divergence는 갖지 못하는 negative value를 가질 수 있기 때문에 variance가 아직 높다. 이를 보완하는 방법은 평균값이 0이면서 원래 근사값과 음의 상관관계를 도입하는 항을 추가하는 것이다. 다음과 같은 답을 제안한다.

$$E_q[\frac{p(x)}{q(x)} - 1] = \sum\limits_x\frac{p(x)}{q(x)}q(x) - 1 = \sum\limits_xp(x) - 1 = 0$$

$p(x)$가 유효한 확률분포로 가정되기 때문에 이렇게 표현할 수 있다. 따라서, 우리는 근사값을 다음과 같이 업데이트 할 수 있다. 

$$approxKL = \frac{p(x)}{q(x)}log\frac{p(x)}{q(x)}$$


## 추가
OpenAI baseline, Stable baseline, TF agent 를 살펴봤으나 
TF-agent 는 ^2 stable baseline은 이거, openai baseline은 ??