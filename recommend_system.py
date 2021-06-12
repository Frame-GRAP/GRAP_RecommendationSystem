#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import yaml
from flask import Flask

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

from surprise.dataset import DatasetAutoFolds
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV


# In[2]:


with open('AWS_config/application.yml',  encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
        AWS_ACCESS_KEY = config["cloud"]["aws"]["credentials"]["accessKey"]
        AWS_SECRET_KEY = config["cloud"]["aws"]["credentials"]["secretKey"]
        AWS_BUCKET = config["cloud"]["aws"]["s3"]["bucket"]
        AWS_region = config["cloud"]["aws"]["region"]["static"]

s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY,
                          region_name=AWS_region
                          )


# # 연관 게임 저장 (content based filtering)

# In[3]:


def save_realated_game():
    data = pd.read_csv('csv/game.csv')
    data = data[['id','genres', 'vote_average', 'vote_count','name','developer', 'publisher']]
    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.6) #상위 n%

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']

        return ( v / (v+m) * R ) + (m / (m + v) * C)

    data['score'] = data.apply(weighted_rating, axis = 1)

    count_vector = CountVectorizer(ngram_range=(1, 3))
    c_vector_genres = count_vector.fit_transform(data['genres'].values.astype('U'))
    c_vector_developer = count_vector.fit_transform(data['developer'].values.astype('U'))

    #코사인 유사도를 구한 벡터를 미리 저장
    gerne_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]
    developer_c_sim = cosine_similarity(c_vector_developer, c_vector_developer).argsort()[:, ::-1]

    def get_recommend_movie_list(df, id, sim, top=30):
        # 특정 게임과 비슷한 게임을 추천해야 하기 때문에 '특정 게임' 정보를 뽑아낸다.
        target_movie_index = df[df['id'] == id].index.values

        #코사인 유사도 중 비슷한 코사인 유사도를 가진 정보를 뽑아낸다.
        sim_index = sim[target_movie_index, :top].reshape(-1)
        #본인을 제외
        sim_index = sim_index[sim_index != target_movie_index]

        result = df.iloc[sim_index][:20]
        return result

    genre_data = get_recommend_movie_list(data, id=1, sim=gerne_c_sim)
    developer_data = get_recommend_movie_list(data, id=1, sim=developer_c_sim)
    result = pd.concat([genre_data,developer_data]).sort_values('score', ascending=False)[:12]

    make_file = open('Json_For_GRAP/related_game_list.json', 'w', encoding='utf-8')
    make_file.write("{")
    j = 0;

    for i in data['id']:
        j = j+1
        genre_data = get_recommend_movie_list(data, id=i, sim=gerne_c_sim)
        developer_data = get_recommend_movie_list(data, id=i, sim=developer_c_sim)
        result = pd.concat([genre_data,developer_data]).sort_values('score', ascending=False)[:12]

        tempdict = {}
        keys = ""

        for key in result['id']:
            keys += str(key) + " "
        tempdict['game_id'] = i
        tempdict['related_game_id'] = keys

        make_file.write('"'+str(i)+'":')
        json.dump(tempdict, make_file, indent="\t")

        if j == len(data['id']):
            break
        make_file.write(',\n')

    make_file.write("}")
    make_file.close()
    s3.upload_file('Json_For_GRAP/related_game_list.json', AWS_BUCKET, 'related_game_list.json')


# # 태그 별 인기게임 저장 (Category tab)

# In[4]:


def save_popular_by_tag():
    data = pd.read_csv('csv/game.csv')
    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.6) #상위 n%

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']

        return ( v / (v+m) * R ) + (m / (m + v) * C)

    data['score'] = data.apply(weighted_rating, axis = 1)
    genre_list = ["Arcade", "Horror", "Action","Adventure", "Casual", "Strategy", "FPS", "RPG"
                    , "Simulation", "Puzzle", "2D", "Atmospheric", "Story Rich", "Sci-fi", "Fantasy", "Colorful"]
    
    make_file = open('Json_For_GRAP/category_tab_list.json', 'w', encoding='utf-8')
    make_file.write("{")
    i = 0;

    for genre in genre_list:
        i = i+1
        tempdict = {}
        keys = ""

        sort_data = data[data['genres'].str.contains(genre) == True].sort_values('score', ascending=False)[:60]

        for key in sort_data['id']:
            keys += str(key) + " "
        tempdict['category_name'] = genre
        tempdict['game_id'] = keys

        make_file.write('"'+str(i)+'":')
        json.dump(tempdict, make_file, indent="\t")
        if i == 16:
            break;
        make_file.write(',\n')

    make_file.write("}")
    
    make_file.close()
    s3.upload_file('Json_For_GRAP/category_tab_list.json', AWS_BUCKET, 'category_tab_list.json')


## 유저를 위한 추천 저장
def save_recommend_for_user():
    data = pd.read_csv('csv/ratings.csv')
    data = data[['userId','movieId', 'rating']]
    data.to_csv('csv/ratings_surprise.csv', index=False, header=False)

    reader = Reader(line_format='user item rating', sep=',',
                   rating_scale=(1, 5))
    data = Dataset.load_from_file('csv/ratings_surprise.csv',reader=reader)
    train, test = train_test_split(data, test_size=0.25,
                                  random_state=42)

    reader = Reader(line_format='user item rating', sep=',',
                   rating_scale=(1, 5))

    data_folds = DatasetAutoFolds(ratings_file='csv/ratings_surprise.csv',
                                 reader=reader)

    trainset = data_folds.build_full_trainset()
    algo = SVD(n_factors=50, n_epochs=10, random_state=42)
    algo.fit(trainset)

    games = pd.read_csv('csv/game.csv')
    ratings = pd.read_csv('csv/ratings.csv')
    ratings = ratings[['userId','movieId', 'rating']]

    movies = pd.read_csv('csv/game.csv')

    def get_unseen_surprise(ratings, movies, userId):
        seen_movies = ratings[ratings['userId']==userId]['movieId'].tolist()
        total_movies = movies['id'].tolist()

        unseen_movies = [movie for movie in total_movies if movie not in seen_movies]  
        return unseen_movies

    def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
        predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

        def sortkey_est(pred):
            return pred.est

        predictions.sort(key=sortkey_est, reverse=True)
        top_predictions = predictions[:top_n]

        top_movie_ids = [int(pred.iid) for pred in top_predictions]
        top_movie_ratings = [pred.est for pred in top_predictions]
        top_movie_titles = movies[movies.id.isin(top_movie_ids)]['name']

        top_movie_preds = [(ids, rating, movies[movies['id']==ids]['id'].iloc[0]) for ids, rating, ids in zip(top_movie_ids, top_movie_ratings, top_movie_ids)]
        return top_movie_preds

    make_file = open('Json_For_GRAP/recommend_for_user_game_list.json', 'w', encoding='utf-8')
    make_file.write("{")
    j = 0;

    max_user_id = ratings.loc[ratings["userId"].idxmax()].iloc[0].astype(int)

    for user_id in range(1,max_user_id+1):
        print(j)
        j = j+1

        unseen_lst = get_unseen_surprise(ratings, movies, user_id)
        top_movies_preds = recomm_movie_by_surprise(algo, user_id, unseen_lst,
                                                   top_n=10)

        tempdict = {}
        keys = ""

        tempdict['user_id'] = user_id

        for top_movie in top_movies_preds:
            keys += str(top_movie[2]) + " "

        tempdict['game_id'] = keys

        make_file.write('"'+str(user_id)+'":')
        json.dump(tempdict, make_file, indent="\t")

        if j == max_user_id:
            break
        make_file.write(',\n')

    make_file.write("}")
    make_file.close()
    s3.upload_file('Json_For_GRAP/recommend_for_user_game_list.json', AWS_BUCKET, 'recommend_for_user_game_list.json')

# # 플라스크 서버

# In[ ]:

app = Flask(__name__)

@app.route("/")
def hello():
    # DB데이터 다운로드
    s3.download_file(AWS_BUCKET, 'game.csv', 'csv/game.csv')
    s3.download_file(AWS_BUCKET, 'ratings.csv', 'csv/ratings.csv')
    
    # 학습 및 계산 후 S3에 저장
    save_popular_by_tag()
    save_realated_game()
    save_recommend_for_user()
    
    # 작업 끝난 뒤 스프링에게 알림
    URL = 'http://localhost:8080/api/util/saveJson'
    requests.get(URL)

    return "done"

if __name__ == "__main__":
    app.run()

