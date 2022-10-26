import numpy as np
import pandas as pd
import tensorflow as tf
from condenser import Condenser
from convectors.layers import OneHot, Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import Activation, Dense, Embedding, Input
from tensorflow.keras.models import Model, load_model

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
MAX_FEATURES = 100000
EMBEDDING_DIM = 500
MAXLEN = 500
SELF_ATTENTION_PARAMS = {
    "units": 128,
    "attention_width": 24,
    "attention_activation": "leaky_relu",
    "attention_type": SeqSelfAttention.ATTENTION_TYPE_MUL
}

df = pd.read_csv("imdb/IMDB Dataset.csv")
df = df.sample(frac=1, random_state=0)
train = df.iloc[:25000]
test = df.iloc[25000:]

# create a preprocessing pipeline using Convectors
nlp = Tokenize(strip_punctuation=False, lower=True)
nlp += Sequence(max_features=MAX_FEATURES, maxlen=MAXLEN)
# process train data
X_train = nlp(train.review)
y_train = np.array([1 if it == "positive" else 0
                    for it in train.sentiment])
# process test data
X_test = nlp(test.review)
y_test = np.array([1 if it == "positive" else 0
                   for it in test.sentiment])
# get number of features
n_features = nlp["Sequence"].n_features + 1

# -----------------------------------------------------------------------------
# Build and fit Keras model
# -----------------------------------------------------------------------------
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
x = SeqSelfAttention(**SELF_ATTENTION_PARAMS)(x)
x = SeqSelfAttention(**SELF_ATTENTION_PARAMS)(x)
x = Condenser(n_sample_points=15,
              reducer_dim=500,
              reducer_activation="tanh",
              characteristic_dropout=.2,
              sampling_bounds=(-100, 100))(x)
out = Dense(1, activation="sigmoid")(x)
model = Model(inp, out)
model.compile("nadam", "binary_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=40, epochs=10,
          validation_data=(X_test, y_test),
          shuffle=True)
