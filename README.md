### README.md

# GRAP 서비스 Recommendation System Repository
- [GRAP_FrontEnd](https://github.com/Frame-GRAP/GRAP_FrontEnd)  
- [GRAP BackEnd](https://github.com/Frame-GRAP/GRAP_BackEnd)  
- [GRAP_Admin](https://github.com/Frame-GRAP/GRAP_Admin) 
----------------------------------------------------------------

## 목차
- [1. 개요](#1-개요)
- [2. Hybrid filtering](#2-hybrid-filtering)
- [3. Cosine Similarity](#3-cosine-similarity) 

## 1. 개요
> 본 시스템은 GRAP 서비스의 게임 추천을 위해 구현된 Flask 기반 Hybrid filtering 추천 시스템입니다.  
> 10,000개의 게임 데이터를 사용하며, 게임 데이터에 Colorful, Horror 등 총 16가지 태그를 각각 붙인 뒤 이를 developer, publisher와 같은 정보와 혼합하여 프로필을 만듭니다. 그 뒤 각 프로필을 벡터화 하고, 서로간의 유사도를 구하는 방법으로 content-based filtering을 적용하고, 이를 총 평가 수 및 평점을 고려한 score 순서대로 정렬하여 유저에게 제공합니다.
> 또한 유저 경험 데이터로 각 유저가 게임을 평가한 데이터 100,000개를 생성하였고, 유저 간의 평가 데이터를 SVD 알고리즘을 사용하여 학습하고 비교분석 하는 방법으로 collaborative filtering을 적용하였습니다. 이는 서비스 내에서 유저 개인 맞춤 추천에 사용됩니다.

## 2. Hybrid filtering

![2](https://user-images.githubusercontent.com/72211590/120937821-9c605b80-c74a-11eb-947a-a960dd5a0688.png)
![1](https://user-images.githubusercontent.com/72211590/120937798-75a22500-c74a-11eb-9745-e459b9a4973a.png)

> - Collaborative filtering은 유저가 서비스를 이용하며 쌓인 행동, 경험, 활동 데이터를 분석하고, 이를 다른 유저의 경험과 비교하여 유저가 겪어보지 않은 아이템에 대해 어느정도의 선호도를 가지는 지 예측하는 기법입니다. 어떤 데이터를 분석하느냐에 따라 정확도가 높은 추천 목록을 제공할 수 있으나, 데이터가 적은 신규 가입자나 신규 아이템에 대해 추천 정확도가 떨어지는 cold-start 문제가 발생합니다.  
> - Content-based filtering은 사용자가 특정 아이템을 선호하는 경우 해당 아이템과 비슷한 프로필을 가진 다른 아이템을 추천해주는 방식입니다. 많은 양의 사용자 경험 데이터가 필요하지 않아 cold-start 문제가 발생하지 않으나, 아이템의 정보 만으로는 해당 유저의 취향을 정확하게 분석하기 힘들어 collaborative-filtering에 비해 추천 정확도가 떨어지는 문제가 발생합니다.
> - Hybrid filtering은 collaborative filtering기의 cold start 문제를 해결하기 위해 신규 유저 및 신규 아이템에 대해서는 상대적으로 유저 데이터를 덜 요구하는 content-based filtering을 사용하고, 이 후 유저의 경험 데이터가 쌓이면 collaborative filtering 기반의 추천을 동반하여 정확성을 높이는 기법입니다.

## 3. Cosine Similarity
![3](https://user-images.githubusercontent.com/72211590/120937823-9d918880-c74a-11eb-98a7-69b4df769c34.png)

> Cosine Similarity는 벡터와 벡터 간의 유사도를 비교할 때 두 벡터 간의 사잇각을 구하여 얼마나 유사한지 수치로 나타냅니다.  
> 본 서비스에서는 content-based filtering에서 게임 프로필 간의 유사도를 구하기 위하여 사용합니다. 각 게임의 태그, 개발자 등의 프로필 정보를 벡터화 한 뒤 이들 사이의 cosine similarity를 계산하여 추천 결과에 반영합니다.
