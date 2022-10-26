import numpy as np
import pandas as pd
from convectors.layers import OneHot, Sequence, Tokenize
from convectors.linguistics import CountFilter

MAX_FEATURES = 100000
MAXLEN = 500


def get_nlp_model(maxlen):
    nlp = Tokenize(strip_punctuation=False, lower=True)
    nlp += Sequence(maxlen=maxlen, max_features=MAX_FEATURES, min_df=2)
    nlp.verbose = False
    return nlp


def load_20NG():
    from sklearn.datasets import fetch_20newsgroups

    # get training data
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    nlp = get_nlp_model(700)
    # process train data
    X_train = nlp(newsgroups_train.data)
    y_train = newsgroups_train.target
    # process test data
    X_test = nlp(newsgroups_test.data)
    y_test = newsgroups_test.target
    return (X_train, y_train), (X_test, y_test)


def load_MR():
    import pandas as pd

    data = pd.DataFrame(
        open("mr/mr.clean.txt").read().split("\n"),
        columns=["text"])
    split = open("mr/split.txt").read().split("\n")
    split = [item.split("\t") for item in split]
    split = pd.DataFrame(split)
    data["index"] = split[0]
    data["dataset"] = split[1]
    data["label"] = split[2].astype(int)
    del split

    train = data[data.dataset == "train"]
    test = data[data.dataset == "test"]

    nlp = get_nlp_model(MAXLEN)
    X_train = nlp(train.text)
    X_test = nlp(test.text)
    y_train = train.label.values
    y_test = test.label.values
    return (X_train, y_train), (X_test, y_test)


def load_ohsumed():
    one_hot = OneHot(verbose=False)
    # get training data
    train = pd.read_csv(f"oh/oh-train-stemmed.csv")
    test = pd.read_csv(f"oh/oh-test-stemmed.csv")

    nlp = get_nlp_model(MAXLEN)
    # process train data
    X_train = nlp(train.text)
    one_hot(test.intent.tolist() + train.intent.tolist())
    y_train = one_hot(train.intent)
    # process test data
    X_test = nlp(test.text)
    y_test = one_hot(test.intent)
    return (X_train, y_train), (X_test, y_test)


def load_imdb():
    df = pd.read_csv("imdb/IMDB Dataset.csv")
    df = df.sample(frac=1, random_state=1)
    train = df.iloc[:25000]
    test = df.iloc[25000:]

    nlp = get_nlp_model(MAXLEN)
    # process train data
    X_train = nlp(train.review)
    y_train = np.array([1 if it == "positive" else 0
                        for it in train.sentiment])
    # process test data
    X_test = nlp(test.review)
    y_test = np.array([1 if it == "positive" else 0
                       for it in test.sentiment])
    return (X_train, y_train), (X_test, y_test)


def load_r8():
    one_hot = OneHot(verbose=False)
    # get training data
    train = pd.read_csv("r8/r8-train-stemmed.csv")
    test = pd.read_csv("r8/r8-test-stemmed.csv")

    nlp = get_nlp_model(MAXLEN)
    # process train data
    X_train = nlp(train.text)
    one_hot(test.intent.tolist() + train.intent.tolist())
    y_train = one_hot(train.intent)
    # process test data
    X_test = nlp(test.text)
    y_test = one_hot(test.intent)
    # get number of features
    return (X_train, y_train), (X_test, y_test)


def load_r52():
    one_hot = OneHot(verbose=False)
    # get training data
    train = pd.read_csv("r52/r52-train-stemmed.csv")
    test = pd.read_csv("r52/r52-test-stemmed.csv")

    nlp = get_nlp_model(MAXLEN)
    # process train data
    X_train = nlp(train.text)
    one_hot(test.intent.tolist() + train.intent.tolist())
    y_train = one_hot(train.intent)
    # process test data
    X_test = nlp(test.text)
    y_test = one_hot(test.intent)
    # get number of features
    return (X_train, y_train), (X_test, y_test)


def load_oh():
    one_hot = OneHot(verbose=False)
    # get training data
    train = pd.read_csv(f"oh/oh-train-stemmed.csv")
    test = pd.read_csv(f"oh/oh-test-stemmed.csv")

    nlp = get_nlp_model(MAXLEN)
    # process train data
    X_train = nlp(train.text)
    one_hot(test.intent.tolist() + train.intent.tolist())
    y_train = one_hot(train.intent)
    # process test data
    X_test = nlp(test.text)
    y_test = one_hot(test.intent)
    # get number of features
    return (X_train, y_train), (X_test, y_test)


def load(dataset):
    if dataset == "20ng":
        return load_20NG()
    elif dataset == "mr":
        return load_MR()
    elif dataset == "oh":
        return load_ohsumed()
    elif dataset == "imdb":
        return load_imdb()
    elif dataset == "r8":
        return load_r8()
    elif dataset == "r52":
        return load_r52()
    else:
        raise ValueError(f"dataset {dataset} unknown")
