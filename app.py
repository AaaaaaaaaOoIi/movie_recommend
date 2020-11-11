# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

app = Flask(__name__)


@app.route('/search', methods=['GET'])
def get_name():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def post_name():
    keyword = request.form['keyword']

    ratings = pd.read_csv('ml-100k/u.data',
                          sep="\t",
                          names=["user_id", "movie_id", "rating"],
                          usecols=range(3)
                          )

    movies = pd.read_csv('ml-100k/u.item',
                         sep="|",
                         names=["movie_id", "movie_title", "release_date"],
                         usecols=range(3),
                         encoding='latin-1'
                         )

    try:
        return render_template("index.html", \
                               title1="映画検索",\
                               search_movie="キーワードを含む映画です。", \
                               movie_name=[movies[movies['movie_title'].str.contains(keyword)].to_html(classes="data")], titles=movies.columns.values)
    except:
        return render_template("index.html", \
                               title1="映画検索", \
                               search_movie="その映画はデータベースにありません。")


@app.route('/', methods=['GET'])
def get():
    return render_template('index.html', \
                           title2='Form Sample(get)', \
                           message='好きな映画の名前を入力して下さい')


# postのときの処理
@app.route('/', methods=['POST'])
def post():
    name = request.form['name']

    ratings = pd.read_csv('ml-100k/u.data',
                          sep="\t",
                          names=["user_id", "movie_id", "rating"],
                          usecols=range(3)
                          )

    movies = pd.read_csv('ml-100k/u.item',
                         sep="|",
                         names=["movie_id", "movie_title", "release_date"],
                         usecols=range(3),
                         encoding='latin-1'
                         )

    movie_ratings = pd.merge(ratings, movies)
    #

    # ピボットテーブルの作成
    ratings_matrix = ratings.pivot_table(index=['movie_id'], columns=['user_id'], values='rating')
    ratings_matrix.fillna(0, inplace=True)
    # ratings_matrix.head()

    # コサイン類似度の計算
    movie_similarity = 1 - pairwise_distances(ratings_matrix.values, metric='cosine')
    np.fill_diagonal(movie_similarity, 0)
    ratings_matrix = pd.DataFrame(movie_similarity)
    # ratings_matrix.head(10)

    movie_name = movies[movies['movie_title'] == name].index.tolist()
    movie_name = movie_name[0]

    movies['similarity'] = ratings_matrix.iloc[movie_name]
    movies.columns = ['movie_id', 'title', 'release_date', 'similarity']
    #movies.sort_values(["similarity"], ascending=False)

    try:
        return render_template('index.html', \
                               title2='Form Sample(post)', \
                               recommend="あなたにおすすめの映画は......", \
                               tables=[movies.sort_values(["similarity"], ascending=False).head(5).to_html(classes='data')], titles=movies.columns.values)

    except:
        return render_template('index.html', \
                               title2='Form Sample(post)',\
                               message='その映画はデータベースにありません。。。。')


if __name__ == '__main__':
    app.run()
