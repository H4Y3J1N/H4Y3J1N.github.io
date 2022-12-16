---
title: Learning To Rank의 기본 - Pointwise, Pairwise, Listwise
date: 2022-10-21 17:00:00 +0800
categories: [Rankig, 추천시스템]
tags: [추천시스템, evaluate, NDCG, pair wise, point wise, list wise]
math: true
mermaid: true
pin: false
published: False
---

## 구현한 추천 모델 성능 평가       
오늘은 추천 시스템 모델의 objective function과 밀접한 관계가 있는 point-wise와 pair-wise , list-wise에 대해 정리한다. 이들은 loss를 정의하는 방법들이다.     

+ 최근에 교육생 클리닉에서 설명해줬던 부분 :       
+ Objective function, loss function의 차이 [포스팅 예정]      

---

# 1. Point wise approaches
Point-wise는 Loss function에서 한번에 하나의 아이템만 고려한다. 다시 말해, 한개의 입력 데이터에 대해 예측된 값과 정답값에 대한 차이만 계산하는 방법이다. MSE(Mean Square Error) loss가 대표적인 예라고 볼 수 있습니다. ALS 모델이 이런 방식으로 학습하며 loss를 줄인다.
Query(=User)가 input으로 들어왔을 때, 이에 대응하는 하나의 Item만 가져와 Score(User, Item) 를 계산하고, 이를 Label score와 비교하고 측정된 Loss를 최소화시키는 방향으로 optimize하는 것이다. 이 방식을 아이템 개수만큼 반복한다.    
각각의 (User, Item) 별 rate를 모두 업데이트한 후, Score을 내림차순으로 정렬한 뒤, 그 인덱스를 그대로 Rank의 개념으로 표현하는 방법론이다.      
### 한계 :
Item - Item 사이의 관계(유저가 아이템 간에 가진 선호 관계 : ex. A보다 B가 좋아)를 학습에 반영하지 못하고, 아이템 하나하나를 개별적으로 학습한 뒤, Top N을 inference 결과로 출력한다. 전체 아이템을 두고 매긴 score이 아니기 때문에, 어떻게 보면 ranking을 prediction 한다는 관점과는 맞지 않을 수 있다.     
### 장점 :
매우 직관적이고 general한 Loss측정 방법이기에 기존의 classical한 regression, classification 모델을 그대로 사용할 수 있다.      
point-wise loss를 사용하는 MF모델 중 SOTA는 eALS이다. BPR 보다 높은 성능을 보인다는데, 경진대회나 프로젝트에서 ALS의 무서운 성능을 체감해 봤기에 조만간 성능 비교 실험을 해 볼 예정이다. 
      
            
      
# 2. Pair wise approaches
한번에 2개의 아이템을 고려해 비교하는 방식이다. 다시 말해 문서 혹은 아이템을 하나의 ‘쌍’으로 활용하는 것이다. 대표적인 모델로는 BPR이 있다.       
Positive /negative item을 각각 하나씩 고려하게 되므로(ex. A가 B보다 좋아) 아이템간 Rank(순위)가 자연스럽게 형성되고, 모델 학습 과정에서 Ranking이 함께 학습된다.      
구체적인 학습 매커니즘은, Ranking의 ground truth와 가장 많은 pair가 일치하는 optimal한 순서를 찾아내는 것이다. 이런 접근 방법의 핵심은 순서가 뒤바뀌는 것을 cost function에 활용하여, 잘못된 순서를 바로잡는 것이다.       
Score를 계산할 때도 (User, Item)이 아닌 (User, Pos Item, Neg Item)이 필요하다. 한편 너무 당연한 얘기지만, 소비자에게는 선호하는(Positive item)것보다 불호하는(Negative item)것이 더 많다. 따라서 자연스럽게 Dataset에도 Pos item보다 Neg item이 훨씬 많은 비율을 차지한다.
따라서 Triplet(User, Pos Item, Neg Item) 을 구성할 때, Pos item이 과도하게 중복되지 않게 하기 위해서 적절한 Sampling 방법을 사용할 필요가 있다.       
현실적으로 ground truth값이 모두 절대적인 데이터를 찾기가 어려워서, 두 Item 사이의 상대적인 relevancy를 학습하는 방식이 많다.
### 한계 :
연산량이 훨씬 많아진다.
### 장점 :
일반적으로 point-wise보다 더 좋은 성능을 보이며(물론 예외상황도 있다), relative order를 예측하는 것이기 때문에 더 Ranking스럽다.   
   
* RankNet : Binary Cross Entropy loss를 사용하여 Pair-wise를 학습한다. score 자체를 예측한다.   
* LambdaRank : 높은 rank에 해당하는 Item에 더 높은 gradients를 주는 방식으로 학습. score 자체를 예측한다.   
* LambdaMART: Grdient Boosting 방법을 활용. LambdaRank보다 더 좋은 성능을 낸다. score의 변화량을 예측한다.   
* LambdaRank, LambdaMART는 List-wise에서도 사용 가능

              
                     
# 3. List wise approaches
Loss function에서 pred row로 주어진 top N개의 아이템을 모두 고려하는 방법론이다. 다시 말해, pair를 넘어서 inference한 아이템 리스트 전체를 ground truth와 한 번에 비교한다. LGBM Ranker 등의 모델이 list wise 방법론을 사용한다.       
Pair wise방법론이 두 개 아이템 간의 상대적 Rank를 학습에 반영한다면, List wise는 전체 아이템 간의 Rank를 학습한다.       
물론 기본 개념 자체는 모든 아이템에 대한 비교지만, 실제적으로 해당 방법은 연산량이 말도 안 되게 커지기 때문에 경진대회나 프로젝트에서는 N개의 아이템에 대한 NDCG 점수로 모델의 성능을 파악한다.       
### 한계 :
모든 방법 중 가장 연산 시간과 복잡도가 높다. 전체 아이템 대상으로는 많은 cost가 요구되므로 비합리적이다.       
### 장점 :
Listwise는 결과의 ‘랭킹’ 즉 순위를 가장 적절하게 나열하는 것 자체를 목적으로 하기 때문에 성능이 좋은 편이다.          
   
* LambdaRank, LambdaMART는 List-wise에서도 사용 가능   
* SoftRank : 각 Item에 대한 rank 확률 분포를 구한다   
* ListNet : Plackett-Luce model를 사용하여 모든 rank 조합(permutation)에 대한 loss를 계산한다   


       

번외 +        
Rating 모델과 Ranking 모델의 차이점은?       
rating은 대개 pointwise 방법론을 차용한다. Rating은 개별 item마다 매겨진 score가 predicted rating이 아니라, ordering으로서의 score이라는 한계를 가진다.       

       
### Reference
[Pointwise vs. Pairwise vs. Listwise Learning to Rank](https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd)





