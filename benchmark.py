import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from condenser import (Condenser, MultiHeadCondenser, SelfAttention,
                       WeightedAttention)
from tensorflow.keras import layers, models

from data import load


def train_model(
    X_train, y_train, X_test, y_test,
    method="condenser", embedding_dim=500, batch_size=40, epochs=10
):
    # infer sizing parameters from
    maxlen = X_train.shape[1]
    n_features = int(max(X_train.max(), X_test.max()) + 1)
    n_labels = int(max(y_train.max(), y_test.max()) + 1)

    # add special token at the start of documents
    if method == "token":
        X_train = np.hstack([np.full((len(X_train), 1), n_features), X_train])
        X_test = np.hstack([np.full((len(X_test), 1), n_features), X_test])
        n_features += 1
        maxlen += 1

    # create model
    input = layers.Input(shape=(maxlen,))
    embedding = layers.Embedding(
        n_features, embedding_dim, mask_zero=True)(input)
    att_1 = SelfAttention()(embedding)
    att_2 = SelfAttention()(att_1)
    if method == "condenser":
        pooling = Condenser(n_sample_points=15,
                            reducer_dim=embedding_dim,
                            reducer_activation="tanh",
                            characteristic_dropout=.2,
                            attention_type="fc",
                            sampling_bounds=(-100, 100))(att_2)
    elif method == "condenser_weighted":
        pooling = Condenser(n_sample_points=15,
                            reducer_dim=embedding_dim,
                            reducer_activation="tanh",
                            characteristic_dropout=.2,
                            attention_type="weighted",
                            sampling_bounds=(-100, 100))(att_2)
    elif method == "multi_head_condenser":
        pooling = MultiHeadCondenser(
            n_heads=2,
            hidden_dim=embedding_dim//4,
            n_sample_points=15,
            reducer_dim=embedding_dim//2,
            reducer_activation="tanh",
            characteristic_dropout=.2,
            sampling_bounds=(-100, 100))(att_2)
    elif method == "average":
        pooling = layers.GlobalAveragePooling1D()(att_2)
    elif method == "max":
        pooling = layers.GlobalMaxPooling1D()(att_2)
    elif method == "weighted":
        pooling = WeightedAttention()(att_2)
    elif method == "token":
        pooling = layers.Lambda(lambda x: x[:, 0, :])(att_2)

    if n_labels == 2:
        output = layers.Dense(1, activation="sigmoid")(pooling)
    else:
        output = layers.Dense(n_labels, activation="softmax")(pooling)

    model = models.Model(input, output)
    model.compile("nadam", "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # train model
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(X_test, y_test),
                        shuffle=True, verbose=False)
    del model  # sanity check
    # get best accuracy
    best_val_acc = max(history.history["val_accuracy"])
    best_val_acc = round(best_val_acc, 4)
    return best_val_acc


def benchmark_model(
    dataset, method, embedding_dim=500, epochs=10, batch_size=40, n_runs=10,
    to_drive=False
):
    if to_drive:
        from google.colab import drive
        try:
            drive.mount('/content/drive')
        except Exception:
            pass

    (X_train, y_train), (X_test, y_test) = load(dataset)
    filename = f"{dataset}_{method}_{embedding_dim}d_batch{batch_size}"
    print(filename)
    results = []
    for run in range(n_runs):
        accuracy = train_model(X_train, y_train, X_test, y_test,
                               method=method, embedding_dim=embedding_dim, batch_size=batch_size, epochs=epochs)
        results.append({
            "model": filename,
            "accuracy": accuracy,
            "run": run
        })
        print("---", accuracy)
    results = pd.DataFrame(results)
    print(round(results.accuracy.mean(), 4))
    print()

    file_path = f"results/{filename}.csv"
    results.to_csv(file_path)
    if to_drive:
        try:
            shutil.copy(file_path, "/content/drive/MyDrive/condenser/")
        except Exception:
            pass


def benchmark_all_models(
    dataset, to_drive=False, epochs=10, batch_size=40, embedding_size=500, n_runs=10
):
    for method in [
        "max",
        "average",
        "weighted",
        "token",
        "condenser",
        "condenser_weighted"
    ]:
        benchmark_model(dataset, method, to_drive=to_drive,
                        embedding_size=embedding_size,
                        epochs=epochs, batch_size=batch_size, n_runs=n_runs)


def benchmark_all(to_drive=True, epochs=10, batch_size=50, n_runs=10):
    for dataset in ["r52", "oh", "mr", "imdb", "r8", "20ng"]:
        benchmark_all_models(dataset, to_drive=to_drive,
                             epochs=epochs, n_runs=n_runs,
                             batch_size=batch_size)
