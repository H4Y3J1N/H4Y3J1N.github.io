---
title: GraphSAGE(Inductive Representation Learning on Large Graphs) 논문 리뷰
date: 2023-02-25 11:30:00 +0800
categories: [논문리뷰, 추천시스템]
tags: [추천시스템, DL, Graph, GraphSAGE, 논문리뷰]
math: true
mermaid: true
published: True
---

본 포스팅에서는 [GraphSAGE 논문](https://arxiv.org/abs/1706.02216)을 리뷰한다.

공부에 도움이 될 추가 자료
1. [CS224W:GraphSAGE Neighbor Sampling](https://youtu.be/LLUxwHc7O4A)
2. [DSBA Paper Review MultiSAGE](http://dsba.korea.ac.kr/seminar/?mod=document&uid=1330)

---

### 그래프 임베딩 분야의 모델들을 추천시스템 분야에서도 사용할 수 있는 원리가 뭘까?
그래프 임베딩 분야에서 제안된 모델들은 추천 시스템 분야에서도 적용될 수 있다. 추천 시스템은 사용자와 아이템을 노드로 갖는 그래프로 표현될 수 있으며, 이 그래프를 이용하여 사용자에게 맞는 아이템을 추천하는 문제를 다룹니다. 이 때, 그래프 임베딩은 각 사용자와 아이템을 벡터 공간에 임베딩하여, 이들 간의 유사도를 계산하거나, 추천 모델에 입력으로 활용할 수 있다.<br>
GraphSAGE와 GAT 같은 그래프 임베딩 모델들은 그래프 데이터를 입력으로 받아, 노드의 임베딩을 생성한다. 이 때, 그래프 데이터에서 노드의 이웃 노드와의 관계를 학습하여, 이웃 노드들과의 상호작용을 고려한 임베딩을 생성할 수 있다. 추천 시스템에서도, 사용자와 아이템 사이의 상호작용을 그래프로 표현할 수 있으며, 이를 이용하여 사용자의 구매 이력, 아이템의 속성 등을 반영한 임베딩을 생성할 수 있다.<br>
또한, 최근에는 그래프 임베딩과 추천 시스템 모델을 결합하는 연구들도 많이 진행되고 있다. 예를 들어, 그래프 신경망과 AutoEncoder 등을 결합하여, 추천 모델의 성능을 높이는 방법이 제안되고 있다. 이러한 연구들은 추천 시스템 분야에서 그래프 데이터를 활용하는 새로운 가능성을 제시하고 있다.<br>
GraphSAGE에서는 대상 노드의 이웃 노드 세트를 먼저 샘플링한 다음 해당 샘플링된 노드를 사용하여 기능 벡터의 집계를 계산하여 노드 임베딩을 학습한다. 이 집계는 노드의 임베딩으로 사용되며, 분류 또는 회귀와 같은 하위 기계 학습 작업에서 사용할 수 있다. 간단히 말하면, 노드는 이웃의 표현의 비선형 변환이다. 이는 아래 이미지로 설명할 수 있다. <br>
<br>

![image](https://user-images.githubusercontent.com/88483620/221349814-e4f56cf3-4653-4e25-97d2-5e76993fb57c.png)


* GraphSAGE의 확률적 트레이닝 프로세스 요약 
1. N개의 노드 중에서 M개의 노드를 랜덤으로 샘플링한다.
2. 샘플링된 각 노드 v에 대해, 복잡성을 줄이기 위해 샘플링 전략을 사용하여 k-홉 이웃을 가져오고, 계산 그래프를 구성한다.
3. 위에서 얻은 정보를 사용하여 우리의 관심 노드의 임베딩을 생성한다.
4. M개 노드 (우리의 경우 자동 인코더를 사용하여 인접 행렬을 재구성)에 대한 재구성 손실을 계산한다.
옵티마이저를 사용하여 그래디언트 업데이트를 수행한다.
<br>


1. Embedding Generation
그래프의 노드 각각에 대한 embedding을 직접 학습하게 되면, 새로운 노드가 추가되었을 때 그 새로운 노드에 대한 embedding을 추론할 수 없다. 따라서 GraphSage는, 노드 임베딩이 아닌 aggregation function을 학습하는 방법을 제안한다.<br>
실제 graph로 표현되는 데이터는 새로운 node들이 실시간으로 추가되는 경우(Evolving Graph)가 매우 많다. 다음 날 10000명의 신규 가입자가 추가되었다면, x+10000명에 대한 node representation을 처음부터 다시 학습해야 한다.<br>
본 논문에서는 fixed graph에 대한 node embedding을 학습하는 transductive learning 방식의 한계점을 지적하고, evolving graph에서 새롭게 추가된 node에 대해서도 inductive node embedding을 산출할 수 있는 프레임워크인 GraphSage를 제안한다. 이름은 SAmple과 aggreGatE를 결합했다. <br>
inductive Node 임베딩 문제는 굉장히 어렵다. 왜냐하면 지금껏 본 적이 없는 Node에 대해 일반화를 하는 것은 이미 알고리즘이 최적화한 Node 임베딩에 대해 새롭게 관측된 subgraph를 맞추는 (align) 작업이 필요하기 때문이다. inductive 프레임워크는 반드시 Node의 Graph 내에서의 지역적인 역할과 글로벌한 위치 모두를 정의할 수 있는 Node의 이웃의 구조적인 특성을 학습해야 한다.

행렬 분해에 기반한 임베딩 접근법과 달리 관측되지 않은 Node에서도 일반화할 수 있는 임베딩 함수를 학습하기 위해 Node Feature(텍스트, Node 프로필 정보, Node degree 등)를 Leverage한다. 학습 알고리즘에 Node Feature를 통합함으로써 우리는 이웃한 Node Feature의 분포와 각 Node의 이웃에 대한 위상적인 구조를 동시에 학습할 수 있다. 풍부한 Feature를 가진 Graph에 집중하여 우리의 접근법은 또한 (Node Degree와 같이) 모든 Graph에 존재하는 구조적인 Feature를 활용할 수 있다. 따라서 본 논문의 알고리즘은 Node Feature가 존재하지 않는 Graph에도 적용될 수 있다.

<br>
노드의 embedding을 구하기 위해서는, 우선 거리(k)를 기준으로 일정 개수의 neighborhood node를 샘플링한다. 그리고 graphsage를 통해 학습된 aggregation function을 통해 주변 노드의 feature로부터 노드의 임베딩을 계산한다. <br>

>>>>> GraphSAGE 에서 aggregator function의 역할은?
GraphSAGE에서 aggregator function은 각 노드의 이웃 노드들의 정보를 모아서 노드의 임베딩을 생성하는 역할을 한다. 즉, 이웃 노드들의 특성을 집계해서 각 노드의 정보를 나타내는 벡터를 만드는 것이다. 각 aggregator function은 특정 깊이(hop)에서 노드의 이웃 노드들로부터 정보를 수집한다. 예를 들어, aggregator function이 2-hop aggregator인 경우, 노드의 이웃 노드와 이웃 노드의 이웃 노드들의 정보를 수집한다. 이렇게 수집한 정보를 기반으로 노드의 임베딩 벡터를 계산한다. 따라서, 각 노드의 임베딩은 그래프 내의 이웃 노드들의 정보에 따라 달라지며, aggregator function은 이러한 노드 임베딩 생성에 중요한 역할을 한다. 

다시 정리하고 넘어가자면, 각 Node에 대한 고유의 임베딩 벡터를 학습하는 대신, 우리는 Node의 지역 이웃으로부터 Feature 정보를 규합하는 Aggregator Function의 집합을 학습한다. <br>
이렇게 추론된 새로운 노드에 대한 임베딩을 downstream task에 활용한다. 특정 노드의 임베딩을 계산할 때, 거리가 K 만큼 떨어져 있는 노드에서부터 순차적으로 feature aggregation을 적용하는 것이다.<br>

>>>>> 왜 거리가 K 만큼 떨어져 있는 노드에서부터 aggregation을 적용하는 것일까? 가까운 노드에서부터 aggregation을 적용하지 않는 이유는?
GraphSAGE에서는 한 노드의 이웃 노드들의 정보를 aggregate 하여 해당 노드의 embedding을 계산한다. 이 때, 거리가 K 만큼 떨어져 있는 노드에서부터 순차적으로 aggregation을 적용하는 이유는 다음과 같다. 우선, 거리가 가까운 이웃 노드들의 정보만을 이용하여 aggregation을 수행하게 되면 해당 노드의 embedding은 이웃 노드들의 정보만을 반영하게 된다. 이에 반해, 거리가 더 멀리 떨어진 이웃 노드들의 정보를 추가적으로 활용하면 해당 노드의 embedding은 보다 다양하고 전체적인 특성을 더 잘 반영할 수 있다. 또한, 거리가 멀어질수록 이웃 노드의 수는 기하급수적으로 늘어나게 된다. 이 때문에, 모든 이웃 노드들의 정보를 동시에 고려하면 계산 복잡도가 증가하여 embedding을 계산하는 데 필요한 시간이 매우 오래 걸리게 된다. 이에 반해, 거리가 K 이상인 노드들의 정보를 순차적으로 적용하면 계산 복잡도를 낮추면서도 보다 다양한 이웃 노드들의 정보를 활용할 수 있다.<br>
<br>

위 내용을 실제로 구현하기 위해서는 batch를 샘플링하는 방법과 node neighborhood에 대한 정의가 필요하다. <br>

# Batch Sampling
graphsage를 학습하는 과정에서는 batch단위로 연산이 이루어져야 한다.
이러한 샘플링 과정이 없으면, 각 Batch의 메모리와 실행 시간은 예측하기 힘들며 계산량 또한 엄청나다.

>>> Machine Learning with Graphs | 2021 | Lecture 17.2 - GraphSAGE Neighbor Sampling 강의 내용
" 그래서, 강조하자면, GraphSAGE의 핵심 아이디어는 다음과 같다. 특정 노드의 임베딩을 계산하기 위해서는 해당 노드의 K-hop 이웃만 알면 된다는 통찰력에서 출발한다. 따라서 노드를 기반으로 한 미니배치 대신 K-hop 이웃에 기반한 미니배치를 생성하면 신뢰성 있는 방식으로 기울기를 계산할 수 있다. 다시 말해서, 노드를 미니배치에 넣는 대신, 계산 그래프 또는 K-hop 이웃을 미니배치에 넣는 것이다."
>> 계산 그래프나 K-hop 이웃을 mini-batch에 어떻게 넣을 수 있다는 걸까? 
미니배치에 계산 그래프나 K-hop 이웃을 넣기 위해서는 그래프를 인접 리스트나 인접 행렬로 나타내고, 미니배치를 생성할 때 이웃 노드들을 샘플링하여 그래프를 생성하는 과정이 필요하다. 이렇게 생성된 그래프는  pytorch-geometric의 NeighborSampler를 사용하여 미니 배치로 처리할 수 있으며 GPU와 함께 사용할 수도 있다. 이 때, 미니배치 내의 모든 그래프는 동일한 크기여야 하며, 그래프 내의 모든 노드는 동일한 수의 이웃 노드를 가져야 한다. 이를 위해서는 적절한 샘플링 전략을 선택하여 이웃 노드를 선택해야 하며, 이웃 노드의 수가 일정하도록 맞추는 것이 중요한다.
>> 그렇다면 그래프 내의 노드 중에서, 이웃을 단 1개만 가진 노드가 있을 경우에는 어떻게 해야 돼?
GraphSAGE에서는 이웃을 가지지 않은 노드를 위해 self-loop를 추가한다. Self-loop는 노드 자신을 자신의 이웃으로 간주하고, 노드의 피처 벡터와 함께 이웃 노드 벡터를 집계하는 데 사용된다. 따라서 self-loop를 추가하면 모든 노드에 대해 일관된 방식으로 이웃 정보를 수집할 수 있다.

1.3. How to Define Neighborhood?
어떻게 특정 node u의 neighborhood N(u)를 정의하는가?
계산 복잡도를 제어하기 위해 각 iteration 마다 uniform random draw 방식으로 정해진 개수의 최 근접 노드를 샘플링한다.
>>> 한 노드에 대해 거리(k)를 기준으로 일정 개수의 neighborhood node를 샘플링할 때, 한 노드가 다른 노드의 K번쨰 이웃인지는 어떻게 판단하는가?
K-hop neighborhood는 해당 노드로부터 K번째 거리에 있는 모든 노드를 의미한다. 이것은 거리 기반 유사도를 사용하여 측정된다. 노드 간 거리를 계산할 때, 일반적으로는 가장 짧은 경로를 따라 계산하게 된다. 그리고 K-hop 이웃을 구할 때에는, 이 최단 경로에 따라서 해당 노드로부터 K번째에 위치한 모든 노드를 구하게 된다.
>>> 그럼 거리 기반 유사도 계산은, 한 노드의 feature 정보를 모두 고려해서 계산하는가?
보통은 한 노드의 feature 정보를 벡터로 표현하고, 이 벡터 간의 거리를 계산하여 거리 기반 유사도를 구한다. 이 때, 노드의 feature 정보는 그래프 구조와 관련된 정보와 함께 고려하여 계산할 수 있다. 예를 들어, 한 노드와 다른 노드 간의 거리를 계산할 때, 이들 간의 연결 관계와 노드 feature 정보를 모두 고려할 수 있다. 이러한 방식으로 계산된 거리 기반 유사도를 이용하여 K-hop 이웃을 샘플링하고, GraphSAGE에서 aggregator function을 이용하여 이웃 정보를 집계한다.

3.1. Embedding Generation (i.e. forward propagation) Algorithm
2. Aggregator function & Learning Graphsage
2.1. Aggregator function
우리의 접근법의 가장 중요한 아이디어는 Node의 지역 이웃으로부터 Feature Information을 통합하는 방법에 대해 학습한다는 것이다.
일단 모델의 파라미터가 모드 학습되었고 고정되어 있다고 가정하고 Embedding Generation 혹은 Propgation 알고리즘에 대해 설명할 것이다. 
첫 번째는 K 개의 Aggregator Function으로 이 함수는 Node 이웃으로부터 정보를 통합하는 역할을 수행한다.
두 번째는 Weight Matrices의 집합으로, 이들은 모델의 다른 layer나 search depth 사이에서 정보를 전달하는데 사용된다. 

[ Algorighm 1  이미지 ]
위에서 확인할 수 있는 직관은 각 iteration 혹은 search depth에서 Node는 그의 지역 이웃으로부터 정보들을 모으고, 이러한 과정이 반복되면서 Node는 Graph의 더 깊은 곳으로부터 정보를 증가시키면서 얻게 된다는 것이다.
알고리즘1은 Graph 구조와 Node Features가 Input으로 주어졌을 때의 임베딩 생성 과정에 대해 기술하고 있다. 아래에서는 Mini-batch 환경에서 어떻게 일반화할 수 있을지 설명할 것이다. 알고리즘1의 바깥 Loop의 각 단계를 살펴보면, hk 는 그 단계에서의 Node의 Representation을 의미한다.

3.3. Aggregator Architectures

aggregator function은 이웃 노드들로부터의 정보를 aggregate하는 역할을 한다. N차원의 격자 구조 데이터를 이용한 머신러닝 예(텍스트, 이미지 등)들과 달리, Node의 이웃들은 자연적인 어떤 순서를 갖고 있지 않다. 하지만 그래프 데이터의 특성 상, 노드의 neighborhood들 간에는 어떤 순서가 없다. 따라서, aggregator function은 symmetric하고 높은 수용력(high representational capacity)을 지님을 동시에 학습 가능해야 한다. 
알고리즘1에서 보았던 Aggregator Function은 반드시 순서가 정해져있지 않은 벡터의 집합에 대해 연산을 수행해야 한다.

Aggregator Funcion의 대칭성은 우리의 신경망 모델이 임의의 순서를 갖고 있는 Node 이웃 feature 집합에도 학습/적용될 수 있게 한다. 본 논문은 이에 대해 3가지 후보를 검증했다.
- Mean Aggregator : 주변 노드의 임베딩과 자기 자신(ego node)의 임베딩을 단순 평균한 후, 선형 변화와 relu를 적용해 줌으로써, 임베딩을 업데이트. 단지 벡터의 원소 평균을 취한 함수이다.
- LSTM Aggregator : LSTM aggregator는 높은 수용력을 가진다는 장점을 갖고 있다. LSTM의 경우 표현력에 있어서 장점을 지니지만 하지만 LSTM 자체는 symmetric한 함수가 아니라는 문제가 있다.  permutation invariant 하지 않다. 따라서 본 연구에서는, 인풋 노드들의 순서를 랜덤하게 조합하는 방식을 취한다. 따라서 본 논문에서는 LSTM을 Node의 이웃의 Random Permutation에 적용함으로써 순서가 없는 벡터 집합에 대해서도 LSTM이 잘 동작하도록 했다.
- Pooling Aggregator : 각 노드의 임베딩에 대해 선형 변환(linear transformation)을 수행한 뒤, element-wise max pooling을 통해 이웃 노드들의 정보를 aggregate하는 방식. 각 이웃의 벡터는 독립적으로 fully-connected된 신경망에 투입된다. 이후 이웃 집합에 Elementwise max-pooling 연산이 적용되어 정보를 통합한다. 이론 상으로 max-pooling 이전에 여러 겹의 layer를 쌓을 수도 있지만, 본 논문에서는 간단히 1개의 layer 만을 사용하였는데, 이 방법은 효율성 측면에서 더 나은 모습을 보여준다.  계산된 각 피쳐에 대해 max-pooling 연산을 적용함으로써 모델은 이웃 집합의 다른 측면을 효과적으로 잡아내게 된다. 물론 이 때 어떠한 대칭 벡터 함수든지 max 연산자 대신 사용할 수 있다. 본 논문에서는 max-pooling과 mean-pooling 사이에 있어 큰 차이를 발견하지 못하였고 이후 논문에서는 max-pooling을 적용하는 것으로 과정을 통일했다.
>>> 추가 정보
Max-Pooling은 주변 노드들 중에서 가장 중요한 feature를 추출하기 때문에 노이즈가 있는 데이터나 sparse한 데이터에서 좋은 성능을 발휘할 수 있다. 하지만 Mean-Aggregating을 사용하면 더 간단하고 빠른 학습을 할 수 있으며, 높은 성능을 발휘하는 경우도 있다. 따라서 GraphSAGE에서는 이 두 방법 중에서 적절한 방법을 선택하는 것이 중요하다고 볼 수 있다.


2.2. Learning Graphsage
본 연구에서 제안하는 것은, 각 노드들의 feature를 aggregate함으로써 "각 노드의 임베딩을 추론할 수 있는 aggregator function의 파라미터를 학습하는 것".
하지만 optimization objective는 기존의 node2vec과 같은 shallow embedding network와 크게 다르지 않다.
K iteration 후의 node representation인 z 
u,u∈V에 대해 손실함수가 계산되며, aggregator function의 파라미터인 Wk ,k∈{1,2,..,K}가 gradient descent를 통해 학습된다.
z v와 z u 는 random walk를 기반으로 이웃으로 설정된 노드 쌍이고, zvn은 z u에 대한 negative node(이웃이 아닌 노드)이다. 즉, 이웃 노드끼리는 유사도가 높은 임베딩을 갖도록, 이웃이 아닌 노드끼리는 유사도가 낮은 임베딩을 갖도록 학습이 이루어지게 된다.


3.2. Learning the Paremeters of GraphSAGE
완전한 비지도 학습 상황에서 유용하고 예측 능력이 있는 Representation을 학습하기 위해 우리는 Graph 기반의 Loss 함수를 Output Represnetation에 적용하고, Weight Matrices  및 Stochastic Gradient Descent를 통해 Aggregator Funciton의 파라미터를 튜닝해야 한다.

Graph 기반의 Loss 함수는 인접한 Node들이 유사한 Representation을 갖도록 하게 하고 서로 멀리 떨어져 있는 Node들은 다른 Representation을 갖게 만든다.

이 때 v 는 고정된 길이의 Random Walk 에서 u 근처에서 동시에 발생한 Node를 의미한다. Pn 은 Negative Sampling Distribution을 Q 는 Negative Sample의 개수를 의미한다.
>>>>>>>> Random Walk 이란?
Random Walk란, 그래프에서 시작 노드에서 시작하여 무작위로 이웃 노드를 선택하여 이동하면서 일정 길이의 경로를 따라가는 것을 말한다. 즉, 무작위로 이웃 노드를 선택하고 이동하는 과정을 반복하여 그래프 상에서 랜덤한 경로를 탐색하는 것이다.
이를 이용해 랜덤하게 샘플링하여 그래프 내의 다양한 노드들을 탐색하고, 이를 통해 그래프 전체를 대표하는 임베딩을 학습하는 방법이 있다. 이때, Random Walk의 길이는 학습하고자 하는 그래프의 특성에 따라 다양하게 설정될 수 있다.

중요한 것은 바로 이것이다 : Graph 기반의 Loss 함수인 zu는 그래프 내 노드들의 임베딩을 학습하기 위한 함수이다. 하지만, 이 함수에서는 이전의 임베딩 방법과 달리 각 노드의 고유한 임베딩을 직접적으로 학습하지 않다. 대신, 그래프에서 추출한 정보를 바탕으로 각 노드의 임베딩을 도출한다. 즉, 노드의 이웃 정보 등을 사용해 그래프 전체에 대한 임베딩을 학습하고, 그 결과를 각 노드에 적용함으로써 각 노드의 임베딩을 구하는 방식이다. 이 방식은 비지도 학습에서 유용하며, 예측 능력이 있는 임베딩을 학습할 수 있도록 도와준다.




4. Experiments
본 논문에서 GraphSAGE의 성능은 총 3가지의 벤치마크 task에서 평가되었다.
(1) Web of Science citation 데이터셋을 활용하여 학술 논문을 여러 다른 분류하는 것
(2) Reddit에 있는 게시물들이 속한 커뮤니티를 구분하는 것
(3) 다양한 생물학적 Protein-protein interaction 그래프 속에서 protein 함수를 구별하는 것


LSTM, Pooling 기반의 Aggregator가 가장 좋은 성능을 보였다. K=2로 설정하는 것이 효율성 측면에서 좋은 모습을 보여주었고, 이웃 개체들을 sub-sampling하는 것은 비록 분산을 크게 만들지만 시간을 크게 단축되기 때문에 꼭 필요한 단계라고 할 수 있겠다.