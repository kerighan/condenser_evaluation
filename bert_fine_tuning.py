import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from condenser import Condenser, WeightedAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Input)
from tensorflow.keras.models import Model

# get training data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# process train data
X_train = tf.constant(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = tf.constant(newsgroups_test.data)
y_test = newsgroups_test.target

# build model
tf_hub = "https://tfhub.dev/tensorflow/"
# url = tf_hub + "bert_en_uncased_preprocess/3"
# url_2 = tf_hub + "bert_en_uncased_L-12_H-768_A-12/4"
url = tf_hub + "bert_en_uncased_preprocess/3"
url_2 = tf_hub + "small_bert/bert_en_uncased_L-2_H-128_A-2/2"

text_input = Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(url)
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(url_2, trainable=False)
outputs = encoder(encoder_inputs)
mask = encoder_inputs["input_mask"]
sequence_output = outputs["sequence_output"]
pooled_output = outputs["pooled_output"]

# 0.7744
x = Condenser(n_sample_points=15,
              sampling_bounds=(-2, 2),
              reducer_dim=200,
              theta_regularizer="l2")(sequence_output, mask=mask)

# x = Dense(48, activation="tanh")(x)
out = Dense(20, activation="softmax")(x)

# create and fit model
model = Model(text_input, out)
model.compile("adam", "sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=20, epochs=10,
          validation_data=(X_test, y_test),
          shuffle=True)
