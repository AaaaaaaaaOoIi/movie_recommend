import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

ratings = pd.read_csv('ml-1m/ratings.dat',
                      sep="::",
                      names=["user_id", "movie_id", "rating"],
                      usecols=range(3),
                      engine='python'
                      )
# ratings.head()
ratings['rating'] = ratings['rating'] - ratings['rating'].mean()
ratings.head()

movies = pd.read_csv('ml-1m/movies.dat',
                     sep="::",
                     names=["movie_id", "movie_title",],
                     usecols=range(2),
                     encoding='latin-1',
                     engine='python'
                     )
# movies.head()

movie_ratings = pd.merge(ratings, movies)
# movie_ratings.head(100)

# 基本統計量の確認
# round(movie_ratings.describe(), 2)

# ピボットテーブルの作成
ratings_matrix = ratings.pivot_table(index=['movie_id'], columns=['user_id'], values='rating')
ratings_matrix.fillna(0, inplace=True)
#ratings_matrix.head()

# コサイン類似度の計算
movie_similarity = 1 - pairwise_distances(ratings_matrix.values, metric='cosine')
np.fill_diagonal(movie_similarity, 0)
ratings_matrix = pd.DataFrame(movie_similarity)
#ratings_matrix.head(10)


# 映画の名前検索
def search_movie(keyword):
    try:
        print(movies[movies['movie_title'].str.contains(keyword)])

    except:
        print("見つかりません")


keyword = str(input("探したい映画の名前の一部を入力してください："))
search_movie(keyword)


# レコメンドシステム
try:
    movie_name = input("好きな映画を入力してください：")
    name = movies[movies['movie_title'] == movie_name].index.tolist()
    name = name[0]

    movies['similarity'] = ratings_matrix.iloc[name]
    movies.columns = ['movie_id', 'title', 'similarity']
    print("あなたの入力した映画に基づいたオススメの映画です", "\n", movies.sort_values(["similarity"], ascending=False)[0:5])

except:
    print("その映画はデータベースにありません。")