---
title: 목적함수(Objective Function)와 비용함수(Cost Function)의 차이
date: 2022-10-30 11:00:00 +0800
categories: [머신러닝, ML]
tags: [cost function, loss function, objective function, 목적함수 비용함수 차이]
mermaid: true
pin: false
published: true
---

목적함수와 손실함수, 비용함수의 차이는 정확히 무엇일까?    
머신러닝의 학습에 관련된 개념이므로 명확히 알아두고 넘어갈 필요가 있다. 오늘은 이에 대해 정리해본다.  
   
  
# Loss Function
data set가 아닌 data point에 대한 loss를 측정할 때 쓰이는 개념이다. MAE가 아닌 AE라고 이해하면 된다.   
   

# Cost Function
일반적으로 모델 최적화(optimization)란 : cost function의 값을 minimize하는 최적 파라미터를 찾는 과정을 말한다.    
Cost Function은 오차를 계산하므로, Loss Function이 순간순간 만들어낸 loss의 총합 또는 평균을 낸다. 이러는 이유는 batch학습마다 평균낸 loss로 최적화 정도를 파악하기 땜둔이다.    
따라서 Cost Function은 Loss Function을 사용하여 정의될 수 있다.   
   
MSE(Mean Squared Error, 평균 제곱 오차), MAE(Mean Absolute Error, 평균 절대 오차), Binary Cross-entropy (a.k.a logloss) 등을 예시로 들 수 있다.   


# Objective function

목적 함수는 말그대로 특정한 목적(Object)을 가지고 모델 학습을 최적화하는 함수이다. 경사 하강법(Gradient Descent)을 사용한 optimization 방식에선 Cost Function ==  Objective function 라고 생각해도 무방하다.    
하지만 MLE(Maximum Likelihood Estimate)와 같이 확률을 최대로 하는 방법을 사용할 경우에는, 다시 말해 `값을 maximize하는 방법`을 사용할 때는 Cost Function이 아닌 `Objective Function` 이라고 지칭해야 한다.    
기본적으로 cost function이나 loss function은 값을 최소화한다는 의미를 가지기 때문이다.    

[Cross Validated](https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing)의 원문을 그대로 옮기면 다음과 같다.     

> Objective function is the most general term for any function that you optimize during training. For example, a probability of generating training set in maximum likelihood approach is a well defined objective function, but it is not a loss function nor cost function (however you could define an equivalent cost function).

* For example:
* MLE is a type of objective function (which you maximize)
* Divergence between classes can be an objective function but it is barely a cost function, unless you define something artificial, like 1-Divergence, and name it a cost