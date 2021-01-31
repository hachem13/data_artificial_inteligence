import pandas as pd 
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
import numpy as np





def loadData():

    plays = pd.read_csv('../data/lastfm/user_artists.dat', sep='\t')
    artists = pd.read_csv('../data/lastfm/artists.dat', sep='\t', usecols=['id','name'])
    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    ap = ap.rename(columns={"weight": "playCount"})
    artists_name = ap['name']
    return list(set(artists_name[:400]))

def recommendation(artist_choice):
    plays = pd.read_csv('../data/lastfm/user_artists.dat', sep='\t')
    artists = pd.read_csv('../data/lastfm/artists.dat', sep='\t', usecols=['id','name'])

    # Merge artist and user pref data
    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    ap = ap.rename(columns={"weight": "playCount"})

    # Group artist by name
    artist_rank = ap.groupby(['name']) \
        .agg({'userID' : 'count', 'playCount' : 'sum'}) \
        .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
        .sort_values(['totalPlays'], ascending=False)

    artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

    # Merge into ap matrix
    ap = ap.join(artist_rank, on="name", how="inner") \
        .sort_values(['playCount'], ascending=False)

    # Preprocessing
    pc = ap.playCount
    play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
    ap = ap.assign(playCountScaled=play_count_scaled)
    #print(ap)

    # Build a user-artist rating matrix 
    ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
    ratings = ratings_df.fillna(0).values

    # Show sparsity
    #density = float(len(ratings.nonzero()[0])) / (ratings.shape[0] * ratings.shape[1]) * 100
    
    

    # Build a sparse matrix
    X = csr_matrix(ratings)

    n_users, n_items = ratings_df.shape


    user_ids = ratings_df.index.values
    artist_names = ap.sort_values("artistID")["name"].unique()



    add_user = [0]*17632
    num = []

    for item in artist_choice:
        artist_index = list(artists.index[artists['name']==item])#.values)
        num.append(artist_index)
        for i in num:
            new_ratings_df = np.vstack((ratings_df, add_user))
            for j in i:
                index = j#[0]
                add_user[index]=1
                #new_ratings_df = np.vstack((ratings_df, add_user))
           #convert array to DataFrame
            ratings_DF= pd.DataFrame(new_ratings_df) 
       
    new_userID = (ratings_DF.shape[0] - 1)
    ratings = ratings_DF.fillna(0).values
    # Build a sparse matrix
    X = csr_matrix(ratings)
    n_users, n_items = ratings_DF.shape
    user_ids = ratings_DF.index.values
    artist_names = ap.sort_values("artistID")["name"].unique()
    Xcoo = X.tocoo()
    data = Dataset()
    data.fit(np.arange(n_users), np.arange(n_items))
    interactions, weights = data.build_interactions(zip(Xcoo.row, Xcoo.col, Xcoo.data)) 
    train, test = random_train_test_split(interactions)
    # model with best parameters
    model = LightFM(k = 1, n = 1, learning_rate = 0.5, learning_schedule = 'adadelta', loss='warp')
    model.fit(train, epochs=10, num_threads=2)
    
    #prediction
    n_users, n_items = X.shape
    #print(X.shape)
    
    scores = model.predict(new_userID, np.arange(n_items))
    top_items = artist_names[np.argsort(-scores)]
    return top_items[0:10]
