import pandas as pd
from condenser import Condenser, SelfAttention
from convectors.layers import OneHot, Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import Activation, Dense, Embedding, Input
from tensorflow.keras.models import Model, load_model

# 0.9822
# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
MAX_FEATURES = 100000
EMBEDDING_DIM = 500
MAXLEN = 400
SELF_ATT_PARAMS = {
    "attention_width": 12,
    "attention_activation": "leaky_relu",
    "attention_type": SeqSelfAttention.ATTENTION_TYPE_MUL
}

# -----------------------------------------------------------------------------
# NLP Pipeline
# -----------------------------------------------------------------------------
# load dataset
data = pd.DataFrame(
    open("mr/mr.clean.txt").read().split("\n"), columns=["text"])
split = open("mr/split.txt").read().split("\n")
split = [item.split("\t") for item in split]
split = pd.DataFrame(split)
data["index"] = split[0]
data["dataset"] = split[1]
data["label"] = split[2].astype(int)
del split

train = data[data.dataset == "train"]
test = data[data.dataset == "test"]
# create a preprocessing pipeline using Convectors
nlp = Tokenize(strip_punctuation=False, lower=True)
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES, min_df=2)
nlp.verbose = False
X_train = nlp(train.text)
X_test = nlp(test.text)
y_train = train.label.values
y_test = test.label.values
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
out = Dense(1, activation="sigmoid")(x)
model = Model(inp, out)
model.compile("nadam", "binary_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=40, epochs=10,
          validation_data=(X_test, y_test),
          shuffle=True)
