from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Concatenate, Multiply, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils.visualize_util import model_to_dot
#from IPython.display import SVG


def get_model(num_movies, num_users, dropout_rate=0.2, deep_layers=[50, 100]):
    latent_dim = 10

    # Define inputs
    movie_input = Input(shape=[1],name='movie-input')
    user_input = Input(shape=[1], name='user-input')

    # MLP Embeddings
    movie_embedding_mlp = Embedding(num_movies + 1, latent_dim, name='movie-embedding-mlp')(movie_input)
    movie_vec_mlp = Flatten(name='flatten-movie-mlp')(movie_embedding_mlp)

    user_embedding_mlp = Embedding(num_users + 1, latent_dim, name='user-embedding-mlp')(user_input)
    user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)

    # MF Embeddings
    movie_embedding_mf = Embedding(num_movies + 1, latent_dim, name='movie-embedding-mf')(movie_input)
    movie_vec_mf = Flatten(name='flatten-movie-mf')(movie_embedding_mf)

    user_embedding_mf = Embedding(num_users + 1, latent_dim, name='user-embedding-mf')(user_input)
    user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)

    # MLP layers
    concat = Concatenate()([movie_vec_mlp, user_vec_mlp])
    layer_dropout = Dropout(dropout_rate)(concat)

    for idx, neurons in enumerate(deep_layers):
        d_l = Dense(neurons, name=f'fc-{idx}', activation='relu')(layer_dropout)
        d_l_bn = BatchNormalization(name=f'batch-norm-{idx}')(d_l)
        layer_dropout = Dropout(dropout_rate)(d_l_bn)


    # Prediction from both layers
    pred_mlp = Dense(10, name='pred-mlp', activation='relu')(layer_dropout)
    pred_mf = Multiply()([movie_vec_mf, user_vec_mf])
    combine_mlp_mf = Concatenate()([pred_mf, pred_mlp])

    # Final prediction
    result = Dense(1, name='result', activation='sigmoid')(combine_mlp_mf)

    model = Model([user_input, movie_input], result)
    return model
