import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


ratings = pd.read_csv('ml-100k/u.data',
                      sep="\t",
                      names=["user_id", "movie_id", "rating"],
                      usecols=range(3)
                      )
# ratings.head()

movies = pd.read_csv('ml-100k/u.item',
                     sep="|",
                     names=["movie_id", "movie_title", "release_date"],
                     usecols=range(3),
                     encoding='latin-1'
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


print("------------------------------------------------------")
keyword = str(input("探したい映画の名前の一部を入力してください："))
print("------------------------------------------------------")
search_movie(keyword)
print("------------------------------------------------------")


# レコメンド
try:
    movie_name = input("好きな映画を入力してください：")
    print("------------------------------------------------------")
    name = movies[movies['movie_title'] == movie_name].index.tolist()
    name = name[0]

    movies['similarity'] = ratings_matrix.iloc[name]
    movies.columns = ['movie_id', 'title', 'release_date', 'similarity']
    print("あなたの入力した映画に基づいたオススメの映画です", "\n", movies.sort_values(["similarity"], ascending=False)[0:5])

except:
    print("その映画はデータベースにありません。")