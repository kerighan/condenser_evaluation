import tensorflow as tf
from condenser import Condenser, SelfAttention
from convectors.layers import Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import (Activation, Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.models import Model, load_model

# 0.8796
# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
MAX_FEATURES = 100000
EMBEDDING_DIM = 500
MAXLEN = 500
SELF_ATT_PARAMS = {
    "attention_activation": "leaky_relu",
    "attention_width": 12,
    "attention_type": SeqSelfAttention.ATTENTION_TYPE_MUL,
}

# -----------------------------------------------------------------------------
# NLP Pipeline
# -----------------------------------------------------------------------------
# get training data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# create a preprocessing pipeline using Convectors
nlp = Tokenize(strip_punctuation=False, lower=True)
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES,
                max_df=.5, min_df=4)

# process train data
X_train = nlp(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = nlp(newsgroups_test.data)
y_test = newsgroups_test.target
# get number of features
n_features = nlp["Sequence"].n_features + 1

# -----------------------------------------------------------------------------
# Build and fit Keras model
# -----------------------------------------------------------------------------
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
# x = SeqSelfAttention(**SELF_ATT_PARAMS)(x)
# x = SeqSelfAttention(**SELF_ATT_PARAMS)(x)
x = SelfAttention()(x)
x = SelfAttention()(x)
x = Condenser(n_sample_points=15,
              reducer_dim=500,
              reducer_activation="tanh",
              characteristic_dropout=.2,
              sampling_bounds=(-100, 100))(x)
out = Dense(20, activation="softmax")(x)

model = Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
# fit model
model.fit(X_train, y_train,
          batch_size=40, epochs=10,
          validation_data=(X_test, y_test))
