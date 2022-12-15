---
title: LGBM 모델에서 lambdarank objective의 내부작동원리
date: 2022-11-14 15:00:00 +0800
categories: [Rankig, 추천시스템]
tags: [추천시스템, LGBM Ranker, lambdarank]
math: true
use_math: true
pin: false
published: True
---

# Main Reference
본 포스트는 [Frank Fineis의 포스트](https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html)의 내용을 번역하고, 이해를 돕기 위해 글쓴이가 해설 및 추가 공부를 통해 세부해설을 덧붙인 버전이다. 매끄러운 문장과 더 쉬운 설명을 위해 직역과 의역을 섞었고, 생략된 문단도 있다.        
기본 논리 전개 틀은 원문의 흐름을 따라감을 사전에 밝힌다.   
    

# 랭킹의 목표 (== Ranking objective)

sort-order 아이템들을 최적의 방식으로 정렬하는 방식은 Rank 학습(or LETOR. 이 포스트에서는 이후로 Rank라고 지칭한다.)라고도 하며, 사람들이 흔히 쓰지 않는 supervised machine learning의 일종이다.    
직접적인 regression 분석은 아니지만, 일부 Rank 솔루션에는 regression이 포함될 수 있다. 또한 정확한 binary classification은 아니지만, 인기 있는 Rank 모델 몇 가지는 이진 분류와 밀접한 관련이 있다. 이 문장은 Ranker는 regression이나 classification과는 별개의 방법론이라는 뜻이다. 이건 LGBM에서도 XBG에서도 마찬가지다.    
일부 Rank 모델은 랭킹을 매기는 데에 있어서, 확률 부여(assigning probabilities)를 포함하기도 한다. 여기서 확률 부여라는 것은 regression 혹은 classification 모델이 산출한 결과를 prob를 말하는 것인데, 이때의 prob는 유저의 아이템 구매 확률, 혹은 선호 확률을 뜻한다.
Rank모델의 성능을 평가하는 evaluating metrics의 경우에도, "정말 잘 Ranking되었는가"의 여부는 standard precision, recall, or RMSE 보다도 [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) and [ERR](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.4509&rep=rep1&type=pdf)이 더 효과적이다.


> 이 글에서는 `lambdarank`의 objective와 핵심 원리를 다루고,
> `LambdarankNDCG` 라는 optimization function에서 정확히 무슨 일이 일어나고 있는지 자세히 알아보려고 한다.
> `LambdarankNDCG`란 LightGBM에서 LamdbaMART를 estimate(추정)하는 데에 쓰는 함수이다.



## Rank 학습이 어려운 이유
Rank가 기존의 supervised learning tasks보다 어려운 가장 큰 이유는, 데이터가 계층적(hierarchical)이기 때문이다.
일반적인 연속 회귀 분석 또는 binary/multiclass/multilabel 레이블 분류 모델에서는, 하나의 input vector 를 하나의 output vector에 매핑한다.
그리고 각 input은 일반적으로 $IID$ 분포로 가정된다. $IID$ 란 독립항등분포로, 확률변수가 여러 개 있을 때 (X 1 , X 2 , ... , X n) 이들이 상호독립적이며, 모두 동일한 확률분포 f(x)를 가진다면 독립항등분포라고 한다.
추천 task의 최종 결과물은 아이템 목록(list)이다. 이 list 내에서 아이템들의 Rank를 매길 때, 측정 단위(항목)는 서로 독립적이지 않다. 오히려 서로 *relative* 하게(==상대적으로) 차등 순위가 매겨진다. 
다시 말하자면, Ranking models이 출력해야 하는 것은 prob(확률)이나 조건부 평균 $E[Y|X]$ 추정치가 아니라 list 내의 아이템들을 최적의 순서로 정렬(Ranking)한 값이다.
세 가지 아이템이 있을 때, 대해 각각  $\{-\infty, 0, \infty\}$ 의 score를 출력하는 Rank 모델은, 동일한 아이템 각각에 대해 점수 $\{-0.01, -0.005, -0.001\}$를 출력하는 Rank 모델과 정확히 동일한 순위(3위, 2위, 1위가 될 것이다)를 제공한다. Ranking이 회귀나 분류와는 다르다는 말의 근거는 바로 이 맥락에 있다. 


Rank를 위한 데이터셋은 기본적으로 첫 번째 colmn에 종속 변수가 있고, 두 번째 column’에 qid’(=그룹 식별자)를 가 있고, 다른 column들에 다른 feature값이 들어 있다.   
따라서 Rank 모델에 필요한 data shape RNN/LSTM 모델의 input으로 넣어야 하는 데이터와 매우 유사한 구조를 가진다 : (samples, time steps, $y$).    
예를 들자면, 아래처럼 single query 내의 여러 아이템으로 구성된다:    


| Y       |    qid  |   feature_0 | feature_1    | feature_2 | feature_3   |
|---------|---------|-------------|--------------|-----------|-------------|
|    0    |    0    |    12.2     |    -0.9      |   0.23    |   103.0     |
|    1    |    0    |    13.0     |   1.29       |   0.98    |   93.5      |
|    2    |    0    |    14.0     |   1.29       |   0.98    |   93.5      |
|    0    |    0    |    11.9     |   1          |   0.94    |   90.2      |
|    1    |    1    |    10.0     |   0.44       |   0.99    |   140.51    |
|    0    |    1    |    10.1     |   0.44       |   0.98    |   160.88    |

<br>


## Pairwise loss starts with binary classification
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

우리는 일반적으로 다음과 같은 logistic function을 통해 $Pr(y_{ij}\|s_{i}, s_{j})$를 표현한다 : $Pr(y_{ij}\|s_{i}, s_{j}} = \frac{1}{1 + e^{-\sigma(s_{i} - s_{j})}$     
왜냐하면 구별하기 쉽고, pairwise model scores를 $s_{i}-s_{j}$를 $(-\infty, \infty)$에서 probability scale인 [0, 1] 사이로 변환할 수 있기 때문이다.    
상수 $\sigma$는 일반적으로 $\sigma=1$을 설정하지만, LightGBM은 이것을 'sigmoid'라는 이름의 하이퍼 파라미터로 명시한다. 따라서 이후로는 sigmoid라고 표기하겠다.

\begin{align}
\ell\ell_{ij} &= y_{ij}\log(\frac{1}{1 + e^{-\sigma (s_{i} - s_{j})}}) + (1 - y_{ij})\log(\frac{e^{-\sigma(s_{i} - s_{j})}}{1 + e^{-\sigma(s_{i} - s_{j})}})
\end{align}
\begin{align}
&= -y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})}) + \log(e^{-\sigma (s_{i} - s_{j})}) - \log(1 + e^{-\sigma (s_{i} - s_{j})}) - y_{ij}\log(e^{-\sigma (s_{i} - s_{j})}) + y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}
\begin{align}
&= (1 - y_{ij})\log(e^{-\sigma (s_{i} - s_{j})}) - y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

If maximizing $\ell \ell_{ij}$ is good, then minimizing $-\ell \ell_{ij}$ must also be good. Machine learning practicioners commonly refer to -1 times the loglikelihood as `logloss`:

\begin{align}
\text{logloss}\_{ij} = (y_{ij}-1)\log(e^{-\sigma (s_{i} - s_{j})}) + y_{ij}\log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

In the literature on pairwise loss for ranking, right here is where I've witnessed a slight of hand: we only need to learn from the cases where $y_{ij} = 1$. This is because negative cases are symmetric; when $Y_{i} > Y_{j}$ (meaning that the pair $(i, j)$ has the label $y=1$), this implies that $(j, i)$ has label $y=0$. Training on tied pairs doesn't help, because the model is trying to discern between relevant and irrelevant items. Therefore, all we really need to keep are the instances of $y_{ij}=1$, and the pairwise logloss expression can be simplified.

\begin{align}
\text{logloss}\_{ij} = \log(1 + e^{-\sigma (s_{i} - s_{j})})
\end{align}

This loss is also known as "pairwise logistic loss," "pairwise loss," and "RankNet loss" (after the siamese neural network used for pairwise ranking first proposed in [[2]](#2)).

<img align="center" src="../../../../images/pairwise_logistic_loss.png" alt="logistic loss" width="600"/>

It's not too difficult to understand: when $Y_{i} > Y_{j}$, the model will achieve small loss when it can predict scores $s_{i} - s_{j} > 0$. The loss will be large when $s_{j} > s_{i}$. The gradient of the loss for any $s_{i} - s_{j}$ is also controlled by $\sigma$. Larger values of $\sigma$ penalize pairwise inconsistencies than values closer to zero.


## LightGBM implements gradient boosting with the lambdarank gradient
LightGBM is a machine learning library for gradient boosting. The core idea behind gradient boosting is that if you can take the first and second derivatives of a loss function you're seeking to minimize (or an objective function you're seeking to maximize), then LightGBM can find a solution for you using gradient boosted decision trees (GBDTs). Gradient boosting is more or less a functional version of [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization), which is why we need the gradient and hessian. During training it builds a sequence of decision trees each fit to the gradient of the loss function when the model is evaluated on the data at the current boosting iteration.

\begin{align}
\frac{\partial \text{logloss}\_{ij}}{\partial s_{i}} &= \frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{i} - s_{j})}}
\end{align}
\begin{align}
&= \frac{-\sigma}{1 + e^{\sigma(s_{i} - s_{j})}} \hspace{10mm} (\text{mult by} \hspace{1mm} \frac{e^{\sigma(s_{i} - s_{j})}}{e^{\sigma(s_{i} - s_{j})}})
\end{align}
\begin{align}
&= \lambda_{ij}
\end{align}

*This* is where the "lambda" in `lambdarank` comes from.

Note a critical shortcoming of the $\lambda_{ij}$ gradients - **they are completely positionally unaware**. The formulation of the pairwise loss that we've used to derive the $\lambda_{ij}$ treats the loss the same whether we've incorrectly sorted two items near the top of the query or near the bottom of the query. For example, it would assign the same loss and gradient to the pairs $(1, 3)$ and $(100, 101)$ when $s_{i} - s_{j}$ and $Y_{i}$ and $Y_{j}$ are the same. But most eCommerce or Google/Bing users spend like 90% of their time near the top of the query, so it's much more important to optimze the items appearing within the first few positions. A possible correction proposed by Burges in 2006 [[4]](#4) was to just scale $\lambda_{ij}$ by the change in NDCG (a positionally-aware ranking metric) that would be incurred if the incorrect pairwise sorting of items $(i, j)$ was used:

\begin{align}
\lambda_{ij} = \log(1 + e^{-\sigma (s_{i} - s_{j})})|\Delta NDCG_{ij}|
\end{align}

This is known as the **lambdarank gradient**. The claim is that by using this form of the gradient within a loss minimization procedure, we end up maximizing NDCG. Well, if this decision to scale the original $\lambda_{ij}$ by $\|\Delta NDCG_{ij}\|$ seems kind of arbitrary to you, you are not alone. Several researchers have taken issue with it, viewing it more as a hack than a true gradient of a real loss function [[7]](#7). Just know that the term `lambdarank` does not refer to a loss function (like some other LightGBM `objective` strings like `"mse"` or `"mape"`), but to an explicit gradient formulation.

Anyway, we have the positionally-aware "gradient" of the pairwise loss function with respect to a single product-product pair $(i, j)$ of differentially-labeled products. LightGBM and GBDTs generally regress decision trees on the loss gradients computed with respect to *individual samples* (or rows of a dataset), not single products within pairs of products.

In order to just get the *gradient with respect to product i*, we have to accumulate the gradient across all pairs of products where $i$ is the more-relevant item, and symmetrically, across all pairs where $i$ is the less-relevant item. Let $I$ refer to the set of item pairs $(i, j)$ where the first item is more relevant than the second item.

\begin{align}
\lambda_{i} = \sum_{j:\{i, j\} \in I}\lambda_{ij} - \sum_{j:\{j, i\} \in I}\lambda_{ij}
\end{align}

We need to take a quick detour. Confusingly, LightGBM (as well as XGBoost) are known as **gradient boosted** tree learning libraries. They actually implement *Newton boosting* [[3]](#3). Gradient boosting is premised on taking a **first**-order Taylor approximation of the loss function about the current estimates of the loss at each step of the tree-estimation process. But you can get better results by taking higher-order approximations to the loss function, and LightGBM uses a **second**-order approximation. In basic gradient boosting, during each boosting iteration we fit a new decision tree directly to $Y = g_{i}^{k}$, where $g_{i}^{k} = \lambda_{i}^{k}$, the gradient of the loss of the model on iteration $k$. But in Newton boosting, the regression involves both the hessian (designated $h_{i}^{k}$) and the gradient:


\begin{align}
\text{tree}\_{k+1} = \arg\min\sum_{i=1}^{n}h_{i}^{k}\big(-\frac{g_{i}^{k}}{h_{i}^{k}} - \ell\ell_{i}^{k}\big)^{2}
\end{align}


Since we've only derived the first derivative of the loss, let's find the second derivative by applying the quotient rule:

\begin{align}
\frac{\partial^{2} \text{logloss}\_{ij}}{\partial s_{i}^{2}} &= \frac{\sigma^{2}e^{-\sigma(s_{j} - s_{i})}|\Delta NDCG_{ij}|}{(1 + e^{-\sigma(s_{j} - s_{i})})^{2}}
\end{align}
\begin{align}
&= \frac{-\sigma}{1 + e^{-\sigma(s_{j} - s_{i})}}|\Delta NDCG_{ij}| \cdot \frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{j} - s_{i})}}
\end{align}
\begin{align}
&= \lambda_{ij}\frac{-\sigma e^{-\sigma(s_{j} - s_{i})}}{1 + e^{-\sigma(s_{j} - s_{i})}}
\end{align}

And we're done with the math! Later on we'll map these components of the gradient and hessian to the actual C++ code used in LightGBM's actual lambdarank objective source code.

#### **Pointwise, pairwise, or listwise?**
A very confusing aspect of the `lambdarank` gradient is that despite being closely related to the gradient of the classic pairwise loss function, a LightGBM `LGBMRanker` model can score *individual* items within a query. It does not expect for two inputs to be provided as ranking input like `rnk.predict(x1, x2)`. Further, the calculations required to derive the gradient, $\frac{\partial \text{logloss}}{\partial s_{i}}$ involves sums over all pairs of items within a query, as if it were a listwise LETOR algorithm.

The fact is that the `lambdarank` LightGBM gradient is based on pairwise classification, but a lambdaMART model involves fitting decision trees to gradients computed g all pairs of differentially-labeled items within a query. Each individual item (each row in the training data) is assigned its own gradient value, and then LightGBM simply regresses trees on those gradients. This is why we can score individual items like `rnk.predict(x1)`:

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

Other researchers have tried to develop better intuitions and better categorizations of LETOR models other than pointwise/pairwise/listwise. The best exploration of this topic I've found is Google's 2019 paper on *Groupwise Scoring Functions* [[5]](#5) which provides the foundation for the popular [Tensorflow Ranking library](https://github.com/tensorflow/ranking). The paper provides the notion of a *scoring function*, which is different than the objective/loss function. A LambdaMART model is a **pointwise scoring function**, meaning that our LightGBM ranker "takes a single document at a time as its input, and produces a score for every document separately."

## How objective functions work in LightGBM
You might be used to interacting with LightGBM in R or Python, but it's actually a C++ library that just happens to have well built-out R and Python APIs. Navigate to `src/objectives` and you'll see the implementations of specific objective/loss functions such as `rank_objective.cpp` in addition to `objective_function.cpp` which implements a objective class [factory](https://en.wikipedia.org/wiki/Factory_method_pattern).

Each individual objective class **must** define a method named `GetGradients` that can update model `score` (aka "loss") values, `gradients`, and `hessians`. Each objective file may contain multiple objective classes. The  `GDBT` class contained within `boosting/gbdt.cpp` that actually [calls](https://github.com/microsoft/LightGBM/blob/master/src/boosting/gbdt.cpp#L178) `GetGradients` to fit regression trees within LightGBM's main training routine.

In `rank_objective.cpp`, `GetGradients` just iterates over queries in the training set, calling an inner `GetGradientsForOneQuery` that calculates the pairwise loss and lambdarank gradients/hessians for each item within each query.

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


## Connecting the math to the code
All of the `lambdarank` math is located primarily in two methods within the `LambdarankNDCG` objective class:
1. [`GetSigmoid`](https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp#L228) performs a look-up operation on a pre-computed, discretized logistic function: \begin{align}
\frac{\lambda_{ij}}{|\Delta NDCG_{ij}|} &= \frac{1}{1 + e^{\sigma(s_{i} - s_{j})}}
\end{align} stored in a vector named `sigmoid_table_`. Using pre-computed logistic function values reduces the number of floating point operations needed to calculate the gradient and hessian for each row of the dataset during each boosting iteration. `GetGradientsForOneQuery` passes $(s_{i} - s_{j})$ to `GetSigmoid`, which applies a scaling factor to transform $(s_{i} - s_{j})$ into an integer value (of type `size_t` in C++) so that can then be used to look up the corresponding value within `sigmoid_table_`.
2. [`GetGradientsForOneQuery`](https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp#L140) processes individual queries. This method launches two `for` loops, the outer loop iterating from `i=0` to `truncation_level_` (a reference to the [`lambdarank_truncation_level`](https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambdarank_truncation_level) parameter) and the inner loop iterating from `j=i+1` to `cnt`, the latter being the number of items within the query at hand. This is where the math comes in:

```C++
// calculate lambda for this pair
double p_lambda = GetSigmoid(delta_score);              // 1 / (1 + e^(sigma * (s_i - s_j)))
double p_hessian = p_lambda * (1.0f - p_lambda);        // Begin hessian calculation.
// update
p_lambda *= -sigmoid_ * delta_pair_NDCG;                // Finish lambdarank gradient: -sigma * |NDCG| / (1 + e^(sigma * (s_i - s_j)))
p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;     // Finish hessian calculation. See derivation below.
```

Let's tie the code together with the math, as I had particularly struggled to understand why `p_hessian =  p_lambda *  (1 - p_lambda)` was valid:

\begin{align}
\text{p_lambda} &= \frac{1}{1 + e^{\sigma(s_{i} - s_{j})}}
\end{align}
\begin{align}
1 - \text{p_lambda} &= \frac{e^{\sigma(s_{i} - s_{j})}}{1 + e^{\sigma(s_{i} - s_{j})}}
\end{align}
\begin{align}
\text{p_lambda}(1 - \text{p_lambda}) &= \frac{e^{\sigma(s_{i} - s_{j})}}{(1 + e^{\sigma(s_{i} - s_{j})})^{2}}
\end{align}
\begin{align}
\Rightarrow \frac{\partial^{2}\text{logloss}}{\partial s_{i}^{2}} &= \sigma^{2}|\Delta NDCG_{ij}|\text{p_lambda}(1 - \text{p_lambda})
\end{align}

And that's just about it! There are some other tweaks that some LightGBM contributors have made, such as the option to "normalize" the gradients across different queries (controlled with the `lambdarank_norm` parameter), which helps prevent the case where one very long query with tons of irrelevant items gets an unfair "build-up" of gradient value relative to a shorter query.


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