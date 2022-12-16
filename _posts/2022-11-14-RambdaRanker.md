---
title: Learning To Rank와 lambdarank objective의 작동원리
date: 2022-11-14 15:00:00 +0800
categories: [Rankig, 추천시스템]
tags: [추천시스템, LGBM Ranker, lambdarank, LTR, Learning To Rank]
math: true
pin: false
published: False
---

이번 포스트에서는 `Learning To Rank(LTR)`에 대해 다뤄보고자 한다.      
Ranking 모델의 object function인 `lambdarank`도 다룬다.    

> 추가 예정 : `LambdarankNDCG` 라는 optimization function에서 정확히 무슨 일이 일어나고 있는지 자세히 알아보려고 한다.
> `LambdarankNDCG`란 LightGBM에서 LamdbaMART를 estimate(추정)하는 데에 쓰는 함수이다.  

## Main Reference    
본 포스트는 [Frank Fineis의 포스트](https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html) 와 마이크로소프트에서 발표한 논문 [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)과 [이혜진님의 포스팅](https://leehyejin91.github.io/post-learning_to_rank_1/) 을 참고해 영어 원문을 번역하고, 이해를 돕기 위해 글쓴이가 해설 및 추가 공부를 통해 세부해설을 덧붙인 버전이다. 매끄러운 문장과 더 쉬운 설명을 위해 직역과 의역을 섞었고, 생략된 문단도 있다.           
     

# So, What is LTR ?
Rating 추천의 목표는 유저가 아이템에 부여한 점수를 정확히 예측하는 것이며, 이 점수가 큰 순서대로 top N의 추천 순위가 결정된다. ALS 모델(MF계열 모델)이 대표적이다.     
반면 ‘랭킹 예측(ranking prediction) 문제’에서 아이템 점수 예측은 중요하지 않다. 여기서는 어떤 아이템을 더 선호하는지가 중요하며, 아이템 - 아이템 간의 선호 관계를 학습(선호강도의 우열을 정확히 추론)하는 것이 LTR의 목표다. LTR은 ‘랭킹 학습’으로, 이것은 Rating 추천과 완전히 다르다.    

![MLR-search-engine-example svg](https://user-images.githubusercontent.com/88483620/207998679-92632199-2384-4531-8fd2-a1ea6f70b501.png)
_이미지의 retrieval을, 추천에서는 'candidate generation'이라고 이해하면 된다_   

위 그림은  [Wikipedia](https://en.wikipedia.org/wiki/Learning_to_rank)의 Learning To Rank 아키텍쳐 도식이다. Index와 강하게 연관되는 작업은 Top-k retrieval이며, Training data와 강하게 연관되는 작업은 Ranking model이다. 다시 말해, Rating을 예측해 만들어낸 retrieval에 다시 한 번 Ranking 모델을 적용해 순위를 매긴 뒤에 결과 페이지로 추천 내용이 serve하는 것이 LTR이다.     
추천에서의 candidate generation은 most popular나 Apriori, co-visitation matrix 등을 활용해 만들어낸다. 하지만 이것을 그냥 무작위로 추천하는 것은 유의미하지 않을 것이다. 수많은 아이템들 사이에서 유저가 특히 더 좋아할 것을 상위에 노출시키는 것이 추천의 중요한 task다.      

   
# Ranking objective
이렇게 sort-order 아이템들을 최적의 방식으로 Ranking하는 방식을 Rank 학습이라고 한다. regression 분석은 아니지만, 일부 Rank 솔루션에는 regression이 포함될 수 있다. 또한 정확한 binary classification은 아니지만, 인기 있는 Rank 모델 몇 가지는 이진 분류와 밀접한 관련이 있다. 어쨌든 중요한 것은 Ranker는 regression이나 classification과는 별개의 방법론이라는 뜻이다. 이건 LGBM에서도 XBG에서도 마찬가지다.    
일부 Rank 모델은 랭킹을 매기는 데에 있어서 확률(assigning probabilities)을 포함하기도 한다. 여기서 확률이란 regression 혹은 classification 모델이 산출한 결과를 prob를 말하는 것인데, 추천에서의 prob는 유저의 아이템 구매 확률, 혹은 선호 확률을 뜻한다.
Rank모델의 성능을 평가하는 evaluating metrics의 경우에도, "정말 잘 Rank되었는가"의 여부는 standard precision, recall, or RMSE 보다 [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) 나 [ERR](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.4509&rep=rep1&type=pdf)이 더 효과적이다.

> 정리하자면 LTR의 목표는 candidate 내에서의 Ranking이다. (Kaggle discussion에서는 이것을 ReRanking이라고도 부른다.)     


## Rank 학습이 낯선 이유   
Rank가 기존의 supervised learning tasks보다 복잡하게 느껴지는 이유는, 데이터가 계층적(hierarchical/선호순위가 존재)이기 때문이다.
일반적인 연속 회귀 분석 또는 binary/multiclass/multilabel 레이블 분류 모델에서는, 하나의 input vector가 하나의 output vector에 매핑한다. 그리고 각 input은 일반적으로 $IID$ 분포로 가정된다. ($IID$ 란 독립항등분포로, 확률변수가 여러 개 있을 때 (X 1 , X 2 , ... , X n) 이들이 상호독립적이며, 모두 동일한 확률분포 f(x)를 가진다면 독립항등분포라고 한다.)     
추천 task의 최종 결과물은 아이템 목록(list)이다. 이 list 내에서 아이템들의 Rank를 매길 때, 측정 단위(아이템)는 서로 독립적이지 않다. 오히려 서로 *relative* 하게(==상대적으로) 차등 순위가 매겨진다. 
다시 말하자면, **Ranking models이 출력해야 하는 것은 prob(확률)이나 조건부 평균 $E[Y|X]$ 추정치가 아니라 list 내의 아이템들을 최적의 순서로 정렬(Ranking)한 값이다.**   
세 가지 아이템이 있을 때, 대해 각각  $\{-\infty, 0, \infty\}$ 의 score를 출력하는 Rank 모델은, 동일한 아이템 각각에 대해 점수 $\{-0.01, -0.005, -0.001\}$를 출력하는 Rank 모델과 정확히 동일한 순위(3위, 2위, 1위가 될 것이다)를 제공한다. Ranking이 회귀나 분류와는 다르다는 말의 근거는 바로 이 맥락에 있다.    
   


# Learning To Rank Dataset & Model Concept   
이제는 예시 데이터를 사용해 더 자세히 알아보자. 데이터셋의 구조는 [이혜진님의 포스트](https://leehyejin91.github.io/post-learning_to_rank_1/)에서 가져온 것이다. 

   
| query(user) |document(item)|   features  | relevance |
|-------------|--------------|-------------|-----------|
|    u1       |        l1    | x1=[u1, l1] |       2   |
|    u1       |        l2    | x2=[u1, l2] |       5   |
|    u2       |        l2    | x3=[u2, l2] |       1   |
|    u2       |        l3    | x4=[u2, l3] |       4   |

<br>

데이터셋은 크게 4가지(로그 데이터를 생성한 유저, 로그 대상 아이템, 유저와 아이템의 관계를 표현한 피처, 검색어와 문서의 관련성을 표현한 라벨) 정보로 구성된다. 예를 들어 검색어 u1과 관련된 아이템은 {l1, l2}이고, (u1, l1)와 (u1, l2)의 관계를 표현한 피처가 각각 x1, x2다. 중요한 것은 relevance인데, 이는 각 유저가 아이템과 얼마나 관련성 있는지 나타내는 값으로, 추천시스템은 relevance를 implicit 혹은 explicit한 여러 방식으로 정의할 수 있다. 일단 이 테이블에서는 평점이라고 정의하겠다.

![LTR_hyejin_table](https://user-images.githubusercontent.com/88483620/208029112-15c85c16-81b1-43ef-92c3-cea3db30ec08.png)
_출처 : 이혜진님 포스트_

위 이미지에서는 왼쪽의 행렬에 주목하라. 이처럼 유저-아이템 평점 행렬을 만들 수 있다면, 랭킹학습을 위한 데이터셋을 쉽게 만들어 볼 수 있다. 데이터셋을 봤으니, 이제는 모델 학습 컨셉과 loss function까지 차근차근 알아보자.      

> 일반적으로 LTR에서는 relevance(score) 예측 모델을 만든다. input은 feature이며, 모델이 relevance(score)를 예측한다는 점에서 scoring function이라고 한다.    

LTR을 다룰 때 빼놓을 수 없는 것이 바로, RankNet, LambdaRank, LambdaMART이다. 우선 Rank loss에 대해 알아보고, 나머지는 글의 뒷부분에서 천천히 알아보자.


# Pairwise loss(==Ranknet Loss) 이해하기

`lambdarank` LightGBM 모델의 objective는 표준 binary classification 모델의 objective를 수정한 것에 불과하다. 그러므로, 본격적인 시작 전 classification로 간단한 리프레시를 해 보겠다.

두 개의 아이템, $i$ 와 $j$가 있다고 해 보자. 이때 $Y_{i} > Y_{j}$, 즉 $i$ 항목이 $j$ 항목보다 더 (유저에게)관련이 있다(=선호가 높다)고 가정한다.    
이 아이템들이 각각 $X_{i}$ 및 $X_{j}$의 특징 벡터로도 표현될 수 있다고 했을 때, $f(\cdot)$ 모델은 pairwise inconsistencies(불일치) 개수를 최소화하여 pairwise classification loss를 최소화한다.    
$Y_{i} > Y_{j}$일 때, 좋은 모델은 $s_{i} = f(X_{i}) > f(X_{j}) = s_{j}$가 되도록 출력 값 $s_{i}, s_{j}$를 제공할 것이다.    
    
    
그렇다면 논리적으로, pairwise loss는 $s_{i} - s_{j} < 0$일 때 크고 $s_{i} - s_{j} > 0$일 때 작아야 한다. 이 차이를 사용하여 pair $(i, j)$가 "$Y=1$" 또는 $(j, i)$가 "$Y=0$"일 확률을 모델링할 수 있다.    
그리고 이 classification model은 Bernoulli likelihood를 최대화(==MLE, Maximum Likelihood Estimation)할 것이다.  $\mathcal{L}$, $Y_{i} > Y_{j}$인 모든 pair $(i,j)$로 구성된 데이터가 주어지면 $\theta = Pr(y\|x)$로 parameter화된 Bernoulli likelihood는 다음과 같이 표현된다.


\begin{align}
\mathcal{L} = \theta^{y}(1 -\theta)^{1-y}, \hspace{5mm} y \in \{0, 1\}
\end{align}

$\log(\cdot)$가 단조 함수이기 때문에 - $5 < 6 \rightarrow \log(5) < \log(6)$라는 fancy한 방법 - Bernoulli likelihood를 최대화하는 것과 log-likelihood를 최대화하는 것은 같은 작업이다.  Log-likelihood, 혹은 $\ell\ell$는 다음과 같이 주어진다.

\begin{align}
\ell\ell = \log(\mathcal{L}) = y\log(\theta) + (1-y)\log(1 - \theta)
\end{align}

우리는 일반적으로 다음과 같은 logistic function을 통해 $Pr(y_{ij}\|s_{i}, s_{j})$를 표현한다 : $Pr(y_{ij}\|s_{i}, s_{j}) = \frac{1}{1 + e^{-\sigma(s_{i} - s_{j})}}$ 
왜냐하면 구별하기 쉽고, pairwise model scores를 $s_{i}-s_{j}$를 $(-\infty, \infty)$에서 probability scale인 [0, 1] 사이로 변환할 수 있기 때문이다.    
상수 $\sigma$는 일반적으로 $\sigma=1$을 설정하지만, LightGBM은 이것을 `sigmoid`라는 이름의 하이퍼 파라미터로 명시한다. 따라서 이후로는 sigmoid라고 표기하겠다.

\begin{align}
\ell\ell_{ij} &= y_{ij}\log(\frac{1}{1 + e^{-\sigma (s_{i} - s_{j})}}) + (1 - y_{ij})\log(\frac{e^{-\sigma(s_{i} - s_{j})}}{1 + e^{-\sigma(s_{i} - s_{j})}})
\end{align}
\begin{align}
&= -y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})}) + \log(e^{-\sigma (s_{i} - s_{j})}) - \log(1 + e^{-\sigma (s_{i} - s_{j})}) - y_{ij}\log(e^{-\sigma (s_{i} - s_{j})}) + y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}
\begin{align}
&= (1 - y_{ij})\log(e^{-\sigma (s_{i} - s_{j})}) - y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

$\ell \ell_{ij}$를 최대화한 효과가 좋다면, $-\ell \ell_{ij}$를 최소화한 효과도 좋아야 한다. 
ML 엔지니어들은 일반적으로 loglikelihood의 -1배를 `logloss`라고 한다 : 

\begin{align}
\text{logloss}\_{ij} = (y_{ij}-1)\log(e^{-\sigma (s_{i} - s_{j})}) + y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

유저의 비선호를 학습할 필요는 없다. 선호만 학습하는 것이 task를 훨씬 단순화하기 때문이다. 따라서 우리는 $y_{ij} = 1$인 경우에만 고려하면 된다.    
조금 더 덧붙이자면, negative cases가 symmetric하기 때문이다 ; $Y_{i} > Y_{j}$일 때($(i, j)$ pair가 label $y=1$을 가짐을 의미), 이는 $(j, i)$이 $y=0$ label을 가짐을 의미한다.     
모델이 relevant한 아이템과 irrelevant한 아이템을 구별하려고 하기 때문에, tied pairs에 대한 훈련은 도움이 되지 않는다. 그러므로, 우리가 정말로 챙겨야 할 것은 $y_{ij}=1$의 instances이며, 그랬을 때 pairwise logloss를 단순화할 수 있다.    

\begin{align}
\text{logloss}\_{ij} = \log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

이 loss는 "pairwise logistic loss", "pairwise loss", "RankNet loss”로도 알려져 있다. [[2]](#2)

![pairwise_logistic_loss](https://user-images.githubusercontent.com/88483620/207800351-28ae70a5-686a-4f93-815e-336678f7d886.png)
_pairwise logistic loss_ 

위 수식은 그리 어렵지 않다 : $Y_{i} > Y_{j}$이고, scores $s_{i} - s_{j} > 0$를 predict할 수 있을 때 모델은 작은 loss값을 가진다.    
* loss는 $s_{j} > s_{i}$일 때 큰 값을 가진다.      
* 모든 $s_{i} - s_{j}$에 대한 gradient loss는 $\sigma$에 의해 제어된다.    
$\sigma$ 값이 클수록, 0에 가까운 값보다 pairwise 불일치(inconsistencies)에 패널티를 준다.    



## LightGBM : lambdarank gradient로 gradient boosting 구현하기
휴! 이제 드디어 lambdarank에 대해 다뤄볼 수 있다.   

Gradient boosting은 [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)의 functional version이기 때문에 gradient(기울기)와 헤세 정렬(hessian)이 필요하다.    
current boosting iteration에서 model이 evaluate되는 경우, Gradient boosting은 학습을 통해 각각의 loss function의 기울기에 맞는 decision trees를 구축한다.   


\begin{align}
\frac{\partial \text{logloss}\_{ij}}{\partial s_{i}} &= \frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{i} - s_{j})}}
\end{align}
\begin{align}
&= \frac{-\sigma}{1 + e^{\sigma(s_{i} - s_{j})}} \hspace{10mm} (\text{mult by} \hspace{1mm} \frac{e^{\sigma(s_{i} - s_{j})}}{e^{\sigma(s_{i} - s_{j})}})
\end{align}
\begin{align}
&= \lambda_{ij}
\end{align}

`lambdarank` 의 *"lambda"* 는 바로 여기서 유래한다.      
$\lambda_{ij}$ gradients의 중요한 단점을 주목하라. **lambda_gradients는 자신이 현재 어디 있는지, 위치를 완전히 알지 못한다**. (Loss Landscape의 관점에서 이 문장을 이해하는 게 좋다.)
$\lambda_{ij}$를 도출하는 데 사용한 pairwise loss는, 쿼리의 상단/하단 둘 중 어디에서 pairs를 잘못 정렬했는지 여부에 차이를 두지 않고, loss값을 동일하게 처리한다.
예를 들어, $s_{i} - s_{j}$와 $Y_{i}$와 $Y_{j}$가 동일할 때 $(1, 3)$와 $(100, 101)$ 쌍에 동일한 손실과 기울기가 할당되는 것이다.    
그러나 대부분의 eCommerce 또는 Google/Bing 사용자는 시간의 약 90%를 쿼리의 상단에서 보내기 때문에, 처음 몇 개의 위치 내에 나타나는 아이템들을 optimze하는 것이 훨씬 더 중요하다.    
2006년에 Burges [[4]](#4)가 제안한 가능한 correction은, $(i,j)$ 항목의 incorrect pairwise을 사용할 때 발생하는 NDCG(a positionally-aware ranking metric)의 변경으로 $\lambda_{ij}$를 스케일링하는 것이었다 : 


\begin{align}
\lambda_{ij} = \log(1 + e^{-\sigma (s_{i} - s_{j})})|\Delta NDCG_{ij}|
\end{align}

이를 **lambdarank gradient**라고 한다. loss minimization procedure 내에서 이러한 형태의 gradient를 사용함으로써 NDCG를 maximizing한다는 주장이다. 원래의 $\lambda_{ij}$를 $\|\Delta NDCG_{ij}\|$만큼 확장하기로 한 이 결정이 혹시 임의적으로 보이는가? 실제로 몇몇 연구자들은 이를 실제 loss function [7](#7)의 진정한 gradient라기보다는 hack으로 간주하며 문제를 제기하기도 했다 [[7]](#7).    
`lambdarank`라는 용어란 loss function(like some other LightGBM `objective` strings like `"mse"` or `"mape”`)을 지칭하는 것이 아니라 explicit한 gradient formulation을 의미한다는 것을 알아두도록 하자.    

어쨌든, 우리는 positionally-aware "gradient”의 pairwise loss function을 가지고 있다. (물론 이 기울기는 differentially-labeled products인 아이템 single product - product pair $(i, j)$에 대한 것이다.)   
LightGBM과 GBDT는 일반적으로 제품 쌍 내의 단일 제품이 아니라 *individual samples*(또는 데이터 세트의 rows)과 관련하여 계산된 loss gradients 에 대해 decision trees를 회귀학습시킨다.

*product i에 대한 기울기*를 얻기 위해, 우리는 모든 아이템 pair에 대해 gradient를 누적한다. (이때 $i$는 모든 pair에 걸쳐 더 relevant한 아이템이고, symmetrical하다.) 여기서 $i$는 less-relevant한 아이템이다.    
$I$는 첫 번째 아이템이 두 번째 아이템보다 더 relevant한 item pairs 집합 $(i,j)$을 참조하도록 한다.

\begin{align}
\lambda_{i} = \sum_{j:\{i, j\} \in I}\lambda_{ij} - \sum_{j:\{j, i\} \in I}\lambda_{ij}
\end{align}

LightGBM(XGBoost)은 *gradient boosted* tree 학습 라이브러리로 알려져 있다. 그것은 실제로 *Newton boosting* [3](#3)을 구현한다. Gradient boosting은 tree-estimation 프로세스의 각 단계에서 loss의 현재 estimation에 대한 loss funtion의 **first** - 1차 Taylor approximation를 취하는 것을 전제로 한다. 그러나 loss function에 higher-order approximations를 취하면 더 나은 결과를 얻을 수 있으며, LightGBM은 **second** - 2차 근사치를 사용한다.
기본 gradient boosting에서 각 부스팅 반복 중에 우리는 새로운 decision tree를 $Y = g_{i}^{k}$에 직접 적합시킨다. 여기서 $g_{i}^{k}$는 iteration $k$에서 모델 loss의 기울기다.    
그러나 Newton boosting에서 regression은 hessian(designated $h_{i}^{k}$)와 기울기를 모두 포함한다 : 


\begin{align}
\text{tree}\_{k+1} = \arg\min\sum_{i=1}^{n}h_{i}^{k}\big(-\frac{g_{i}^{k}}{h_{i}^{k}} - \ell\ell_{i}^{k}\big)^{2}
\end{align}


loss의 첫 번째 도함수만 도출했으므로, quotient rule을 적용하여 두 번째 도함수를 찾아보자 : 

\begin{align}
\frac{\partial^{2} \text{logloss}\_{ij}}{\partial s_{i}^{2}} &= \frac{\sigma^{2}e^{-\sigma(s_{j} - s_{i})}|\Delta NDCG_{ij}|}{(1 + e^{-\sigma(s_{j} - s_{i})})^{2}}
\end{align}
\begin{align}
&= \frac{-\sigma}{1 + e^{-\sigma(s_{j} - s_{i})}}|\Delta NDCG_{ij}| \cdot \frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{j} - s_{i})}}
\end{align}
\begin{align}
&= \lambda_{ij}\frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{j} - s_{i})}}
\end{align}

수식 계산은 딱 여기까지다!     


#### **Pointwise, pairwise, or listwise?**
내가 이전에 작성한 [Learning To Rank의 기본 - Pointwise, Pairwise, Listwise]()라는 포스트에서 위 3가지 loss 정의 방법에 대해 자세히 다루고 있다.


`lambdarank` gradient의 매우 혼란스러운 측면은, classic한 pairwise loss function의 gradient와 밀접한 관련이 있음에도 불구하고, LightGBM `LGBMRanker` 모델이 쿼리 내에서 *개별* 항목에 score를 매길 수 있다는 것이다. ranking 제시를 위해 'rnk.predict(x1,x2)'처럼 두 개의 inputs을 넣을 필요가 없다.
또한, gradient $\frac{\partial \text{logloss}}{\partial s_{i}}$를 도출하는 데 필요한 계산은, 이것이 마치 listwise Rank 알고리즘인 것처럼 쿼리 내의 모든 아이템 pairs에 대한 합계를 포함한다.    
    
팩트는  `lambdarank` LightGBM gradient 가 pairwise classification에 기초한다는 것이다.
그러나  lambdaMART model모델은 decision tree 모델 학습에도 포함된다. 쿼리 내에서 differentially-labeled된 모든 아이템 pair의 기울기 계산을 위해서다. 
각 개별 아이템(each row in the training data)에 기울기 값이 할당된 다음, LightGBM은 해당 gradients에 대해 tree 모델을 회귀 학습시킨다.    
이것이 우리가 `rnk.predict(x1)`와 같은 개별 아이템에 score를 매길 수 있는 이유이다 : 


```{python}
import lightgbm as lgb
import numpy as np

np.random.seed(123)
X = np.random.normal(size=(100,5))
y = np.random.choice(range(4), size=100, replace=True)
grp = [10] * 10

rnk = lgb.LGBMRanker(objective='lambdarank')  # lambdarank is actually default objective for LGBMRanker
rnk.fit(X, y, group=grp)

rnk.predict(X[50, :].reshape(1, -1))  # pointwise score for row 50
```

> array([-1.95225947])


## How objective functions work in LightGBM

각 objective class는 반드시 모델 'score'(일명 "loss") 값, 'gradients' 및 'hessians'을 업데이트할 수 있는 'GetGradients'라는 이름의 method를 정의해야 한다.
각각의 objective 파일은 여러 개의 objective classes를 포함할 수 있다.
`GDBT` class 는 `boosting/gbdt.cpp` 를 포함하고 있다. 그것은 GetGradients를 실제로 호출해 LightGBM의 메인 학습 루틴에 regression trees를 훈련시킨다.

```
void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    for (data_size_t i = 0; i < num_queries_; ++i) {
        const data_size_t start = query_boundaries_[i];
        const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
        GetGradientsForOneQuery(i, cnt, label_ + start, score + start,
                                gradients + start, hessians + start);
        ...
    }
}

virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                     const label_t* label,
                                     const double* score, score_t* lambdas,
                                     score_t* hessians) const override {
    // score, lambdas, and hessians are modified in-place...
}
```

There are actually a couple of different ranking objectives offered by LightGBM that each subclass a `RankingObjective` wrapper class:
- `LambdarankNDCG`: this is the selected objective class when you set `LGBMRanker(objective="lambdarank")`.
- `RankXENDCG`: Rank-cross-entropy-NDCG loss ($XE_{NDCG}$ for short) is a new attempt to revise the lambdarank gradients through a more theoretically sound argument that involves transforming the model scores $s_{i}$ into probabilities and deriving a special form of multiclass log loss [[6]](#6).



## References
<a id="1">[1]</a>
Joachims, 2008. [SVMlight](https://www.cs.cornell.edu/people/tj/svm_light/)

<a id="2">[2]</a>
Burges, et al., 2005. [Learing to Rank using Gradient Descent](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf)

<a id="3">[3]</a>
Sigrist, 2020. [Gradient and Newton Boosting for Classification and Regression](https://arxiv.org/pdf/1808.03064.pdf)

<a id="4">[4]</a>
Burges, 2010. [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf)

<a id="5">[5]</a>
Ai, Wang, Bendersky et al., 2019. [Learning Groupwise Scoring Functions Using Deep Neural Networks](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/a995c37352b4b7d13923ca945cdcd03227c9023f.pdf)

<a id="6">[6]</a>
Bruch, 2021. [An Alternative Cross Entropy Loss for Learning-to-Rank](https://arxiv.org/pdf/1911.09798.pdf)

<a id="7">[7]</a>
Wang, Li, et al., 2018. [The LambdaLoss Framework for Ranking Metric Optimization](https://research.google/pubs/pub47258/)