import pandas as pd
import tensorflow as tf
from condenser import Condenser, SelfAttention
from convectors.layers import OneHot, Sequence, Tokenize
from keras_multi_head import MultiHead
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import (Activation, Dense, Dropout, Embedding,
                                     Input, LayerNormalization, Reshape)
from tensorflow.keras.models import Model, load_model

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
MAX_FEATURES = 100000
EMBEDDING_DIM = 500
MAXLEN = 500
SELF_ATT_PARAMS = {
    "attention_width": 12,
    "attention_activation": "leaky_relu",
    "attention_type": SeqSelfAttention.ATTENTION_TYPE_MUL
}

# -----------------------------------------------------------------------------
# NLP Pipeline
# -----------------------------------------------------------------------------
one_hot = OneHot(verbose=False)
# get training data
train = pd.read_csv("r8/r8-train-stemmed.csv")
test = pd.read_csv("r8/r8-test-stemmed.csv")

# create a preprocessing pipeline using Convectors
nlp = Tokenize(strip_punctuation=False, lower=True)
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES, min_df=2)

# process train data
X_train = nlp(train.text)
one_hot(test.intent.tolist() + train.intent.tolist())
y_train = one_hot(train.intent)
# process test data
X_test = nlp(test.text)
y_test = one_hot(test.intent)
# get number of features
n_features = nlp["Sequence"].n_features + 1
print(n_features)

# -----------------------------------------------------------------------------
# Build and fit Keras model
# -----------------------------------------------------------------------------
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
# x = SeqSelfAttention(**SELF_ATT_PARAMS)(x)
# x = SeqSelfAttention(**SELF_ATT_PARAMS)(x)
# x = SelfAttention(units=500, attention_width=24,
#                   attention_activation="leaky_relu",
#                   hidden_activation="tanh",
#                   use_positional_encoding=True)(x)
# x = SelfAttention(units=500, attention_width=24,
#                   attention_activation="leaky_relu",
#                   hidden_activation="tanh",
#                   use_positional_encoding=True)(x)
x = SelfAttention()(x)
x = SelfAttention()(x)
x = Condenser(n_sample_points=15,
              reducer_dim=500,
              reducer_activation="tanh",
              characteristic_dropout=.2,
              sampling_bounds=(-100, 100))(x)
out = Dense(one_hot.n_features, activation="softmax")(x)
model = Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
# fit model
model.fit(X_train, y_train,
          batch_size=40, epochs=10,
          validation_data=(X_test, y_test),
          shuffle=True)
