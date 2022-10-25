<!-- ---
layout: post
title: "[RL] 1. Overview"

category: study
sitemap: false
hide_last_modified: true
---

이번 글에서는 들을 살펴보겠습니다. 다음은 스탠포드 대학교의 2019년 강좌 [CS234](https://web.stanford.edu/class/cs234/CS234Win2019/index.html)와 Sutton and Bartod의 [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)을 토대로 정리한 내용입니다.

--- 
- Markov Process
- Markov Reward Process(MRP)
- Markov Decision Process(MDP)
- Evaluation and Control in MDPs -->
<!-- 

강화학습에 대한 관심
대부분의 머신러닝 알고리즘은 사람들의 데이터를 통한 지도학습을 실시한다.
따라서 '사람처럼 잘'해도 '사람보다 잘'하기는 쉽지않다. 

지도학습을 기반으로하지만 사람보다 뛰어난 성능을 보이는 경우는 대체로 두가지로 나뉜다.
한 사람이 방대한 데이터를 접하기 어려운 태스크인 경우, 사람들이 종종 하는 실수를 데이터의 양으로 극복하는 경우.

예를들어 영화, 상품, 음악 등의 추천시스템인 경우 각 분야의 전문가일지라도 방대한 양의 콘텐츠나 상품을 접하기가 쉽지않다.
따라서 사람이 직접 추천해주는 것보다 데이터를 기반한 알고리즘이 훨씬 좋은 성능을 가진다. 정형 데이터를 사용하는
다양한 예측 업무도 마찬가지다. 한 사람이 표에 있는 수십수백이 넘는 지표들과 특성들을 종합적으로 고려해서 최적의 판단을 
내리는 것은 어렵다. 그래서 더 잘한다. -->