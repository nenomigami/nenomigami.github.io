# Data-Efficient Hierarchical Reinforcement Learning

## Abstract
Hierarchical reinforcement learning (HRL) is a promising approach to extend traditional reinforcement learning (RL) methods to solve more complex tasks.

Hierarchical reinforcement learning은 더 복잡한 문제를 풀기위해 전통적인 RL방법을 확장한 전망있는 접근법이다. 

Yet, the majority of current HRL methods require careful task-specific design and on-policy training, making them difficult to apply in real-world scenarios. 

그러나, 대부분의 현재 HRL 방법들은 섬세한 task-specific 디자인을 요구하며, on-policy traning 이다. 이는 실생활 적용을 어렵게 만들고 있다.

In this paper, we study how we can develop HRL algorithms that are general, in that they do not make onerous additional assumptions beyond standard RL algorithms, and efficient, in the sense that they can be used with modest numbers of interaction samples, making them suitable for real-world problems such as robotic control. 

이 논문에서, 우리는 스탠다드 RL 알고리즘보다 부담되는 추가적인 가정을 넣지 않음으로써 더 일반적인 HRL 알고리즘을 만드는 법을 연구했고, 로봇 컨트롤과 같은 실생활 문제에 적절하게끔 보통의 숫자의 상호작용 샘플과 사용할 수 있게 효율적인 알고리즘을 만드는 것을 연구했다.

For generality, we develop a scheme where lower-level controllers are supervised with goals that are learned and proposed automatically by the higher-level controllers.

일반성을 위해, lower-level controllers가 higher-level controller 에 의해 학습되고 제안된 목표를 받는 구조를 개발했다.

To address efficiency, we propose to use off-policy experience for both higher and lower-level training. 

효율성을 위해, 우리는 off-policy 경험을 higher, lower 둘다 적용했다.

This poses a considerable challenge, since changes to the lower-level behaviors change the action space for the higher-level policy, and we introduce an off-policy correction to remedy this challenge. 

이는 상당한 어려움을 불렀다. 왜냐하면 lower-level 행동의 변화는 high-level policy의 action space 를 변화시키기 때문이다. 따라서 우리는 off-policy correction 기법을 도입했다.

This allows us to take advantage of recent advances in off-policy model-free RL to learn both
higher and lower-level policies using substantially fewer environment interactions than on-policy algorithms. 

이는 on-policy algorithm보다 상당히 적은 양의 상호작용을 사용하여 higher, lower polices 둘을 학습하는데 최근 off-policy model-free RL의 최근의 많은 발전의 이점을 이용할 수있게 해주었다. 

We term the resulting HRL agent HIRO and find that it is generally applicable and highly sample-efficient.

우리는 HRL agent 를 HIRO 로 명명했으며 이것이 일반적으로 적용되며 굉장히 sample-efficinet 함을 발견했다.

Our experiments show that HIRO can be used to learn highly complex behaviors for simulated robots, such
as pushing objects and utilizing them to reach target locations,1 learning from only a few million samples, equivalent to a few days of real-time interaction. 

우리의 실험은 HIRO가 object를 밀거나, 활용해서 목표지점까지 당도하는데 등 로봇 시뮬레이션의 상당히 복잡한 행동을 학습하는데 사용할 수 있음을 보여주었다. 현실 상호작용으로 며칠정도에 해당하는 수백만개의 샘플으로도 학습할 수 있었다.

In comparisons with a number of prior HRL methods, we find that our approach substantially outperforms previous state-of-the-art techniques.

이전 HRL 몇몇 방법과 비교하면, 우리의 접근법이 이전 SOTA보다 상당히 앞섰음을 확인할 수 있다.


## 2 Introduction
Deep reinforcement learning (RL) has made significant progress on a range of continuous control
tasks, such as locomotion skills [39, 27, 18], learning dexterous manipulation behaviors [36], and
training robot arms for simple manipulation tasks [13, 46]. 

Deep RL은 이동기술이나, 손재주 기술, 간단한 조작을 위한 로봇 팔 기술 등 continous control 측면에서 상당한 진보를 이루고있다.

However, most of these behaviors are inherently atomic: they require performing some simple skill, either episodically or cyclically, and rarely involve complex multi-level reasoning, such as utilizing a variety of locomotion behaviors to accomplish complex goals that require movement, object interaction, and discrete decision-making. 

그러나, 대부분의 행동들은 본질적으로 너무 단순하다. episodically 혹은 cyclically 하게끔 간단한 기술만을 시연하기를 요구한다. 이동, 오브젝트 상호작용, 이산적인 의사결정 등 복잡한 목표를 성취를 위한 다양한 이동기술을 활용하는 복잡한 다단계 사고과정이 전혀 들어있지 않다.

Hierarchical reinforcement learning (HRL), in which multiple layers of policies are trained to perform
decision-making and control at successively higher levels of temporal and behavioral abstraction, has
long held the promise to learn such difficult tasks [7, 32, 43, 4]. 

HRL 에서는 다수의 정책 layer가 점점 더 높은 레벨의 시간적,행동적으로 추상화된 의사결정과 control을 수행하기 위해 학습된다. 

By having a hierarchy of policies, of which only the lowest applies actions to the environment, one is able to train the higher levels to plan over a longer time scale. 

가장 낮은 정책만이 action을 envirionment에 적용할 수 있는 policy 계층을 가짐으로써, action 하나가 더 긴 time scale 를 계획하는 higher level 까지 훈련할 수 있다.

Moreover, if the high-level actions correspond to semantically different low-level behavior, standard exploration techniques may be applied to more appropriately explore a complex environment.

게다가, high-level 액션들이 의미론적으로 다른 low-level들과 대응되면, 일반적 탐색 방법들은 아마 더 적절하게 복잡한 환경을 탐험할 것이다.

Still, there is a large gap between the basic definition of HRL and the promise it holds to successfully solve complex environments. 

그러나, HRL 의 기본 정의와 복잡한 문제를 풀 가능성은 상당한 차이가 있다.

To achieve the benefits of HRL, there are a number of questions that one must suitably answer: How should one train the lower-level policy to induce semantically distinct behavior? How should the high-level policy actions be defined? How should the multiple policies be trained without incurring an inordinate amount of experience collection? Previous work has attempted to answer these questions in a variety of ways and has provided encouraging successes [48, 10, 11, 19, 40]. 

HRL의 이점을 성취하기위해서, 적절히 대답해야하는 질문들이 몇 가지 있다. 
- 어떻게 lower-level policy 를 훈련해서 의미론적으로 구별되는 행동을 유도할 수 있는가
- 어떻게 high-level policy 의 actions 이 정의될 것인가
- 어떻게 다수의 정책들이 지나친 경험 집합을 만들지 않고 훈련될 것인가
이전의 연구들은 이런 질문에 다양한 방법으로 답하려 시도했으며 의미있는 성과를 남겼다.

However, many of these methods lack generality, requiring some degree of manual task-specific design, and often require expensive on-policy training that is unable to benefit from advances in off-policy model-free RL, which in recent years has drastically brought down sample complexity requirements [12, 16, 3].

그러나, 대부분의 방법들은 범용성이 부족하며, 어느정도 수동적인 조작이 필요했으며, 종종 expensive 한 on-policy 훈련을 요구하여 최근에 비약적으로 sample complexity 요구량을 줄인 off-policy model-free RL의 이점을 못살렸다.

For generality, we propose to take advantage of the state observation provided by the environment
to the agent, which in locomotion tasks can include the position and orientation of the agent and its
limbs. 

범용성을 위해, 우리는 

We let the high-level actions be goal states and reward the lower-level policy for performing
actions which yield it an observation close to matching the desired goal. 

In this way, our HRL setup does not require a manual or multi-task design and is fully general.

This idea of a higher-level policy commanding a lower-level policy to match observations to a goal
state has been proposed before [7, 48]. 

Unlike previous work, which represented goals and rewarded matching observations within a learned embedding space, we use the state observations in their raw form. 

This significantly simplifies the learning, and in our experiments, we observe substantial benefits for this simpler approach.

While these goal-proposing methods are very general, they require training with on-policy RL
algorithms, which are generally less efficient than off-policy methods [15, 31]. 

On-policy training has been attractive in the past since, outside of discrete control, off-policy methods have been plagued with instability [15], which is amplified when training multiple policies jointly, as in HRL. 

Other than instability, off-policy training poses another challenge that is unique to HRL. Since the lower-level policy is changing underneath the higher-level policy, a sample observed for a certain high-level action in the past may not yield the same low-level behavior in the future, and thus not be a valid experience for training.

This amounts to a non-stationary problem for the higher-level policy. We remedy this issue by introducing an off-policy correction, which re-labels an experience in the past with a high-level action chosen to maximize the probability of the past lower-level actions. 

In this way, we are able to use past experience for training the higher-level policy, taking advantage of progress made in recent years to provide stable, robust, and general off-policy RL methods [12, 31, 3].
In summary, we introduce a method to train a multi-level HRL agent that stands out from previous
methods by being both generally applicable and data-efficient. 

Our method achieves generality by training the lower-level policy to reach goal states learned and instructed by the higher-levels. 

In contrast to prior work that operates in this goal-setting model, we use states as goals directly, which allows for simple and fast training of the lower layer. Moreover, by using off-policy training with our novel off-policy correction, our method is extremely sample-efficient.

We evaluate our method on several difficult environments. These environments require the ability to perform exploratory navigation as well as complex sequences of interaction with objects in the environment (see Figure 1).

While these tasks are unsolvable by existing non-HRL methods, we find that our HRL setup can learn successful policies. 

When compared to other published HRL methods, we also observe the superiority of our method, in terms of both final performance and speed of learning. 

In only a few million experience samples, our agents are able to adequately solve previously unapproachable tasks.