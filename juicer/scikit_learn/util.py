import numpy as np
from gettext import gettext

def get_X_train_data(df, features):
    """
    Method to convert some Pandas's columns to the Sklearn input format.
    Many sklearn ML algorithm receives an array-like of shape
    (n_samples, n_features)

    :param df: Pandas DataFrame;
    :param features: a list of columns;

    :return: a DataFrame's subset as a list of list.
    """

    # Finding columns of list type (e.g., data produced by OneHotEncode)
    # to convert in a 2D-array
    column_list = []
    for feature in features:
        if isinstance(df[feature].iloc[0], list):
            column_list.append(feature)

    columns = [col for col in features if col not in column_list]
    tmp1 = df[columns].to_numpy()

    if len(column_list) > 0:
        tmp2 = df[column_list].sum(axis=1).to_numpy().tolist()
        output = np.concatenate((tmp1, tmp2), axis=1)
    else:
        output = tmp1
    return output.tolist()


def get_label_data(df, label):
    """
    Method to check and convert a Panda's column as a Python built-in list.

    :param df: Pandas DataFrame;
    :param labels: a list of columns;

    :return: A column as a Python list.
    """

    # Validating multiple columns on label
    if len(label) > 1:
        raise ValueError(gettext('Label must be a single column of dataset'))

    # Validating OneHotEncode data existence
    if isinstance(df[label[0]].iloc[0], list):
        raise ValueError(gettext('Label must be primitive type data'))

    y = df[label].to_numpy().tolist()
    return np.reshape(y, len(y))

def soundex(token: str):
    # Convert the word to upper case for uniformity
    token = ''.join([c for c in token if c.isalpha()]).upper()

    soundex = token[0]

    token = token.translate(
        str.maketrans(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "01230120022455012623010202"))
    #import pdb; pdb.set_trace()
    token = ''.join([token[i] for i in range(1, len(token))
                     if token[i] != token[i-1] and token[i] != '0'])

    # Append zeros to the end of the string until it has a length of 4
    soundex += token + "0" * (4 - len(token))

    return soundex[:4]


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def to_json(s):
    class CustomParser(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pl.Series):
                return obj.to_list()
            return json.JSONEncoder.default(self, obj)

    return json.dumps(s, cls=CustomParser)


def levenshtein(s1, s2):
    # If either string is empty, the distance is the length of the other string
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Initialize the distance matrix
    distance = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
    for i in range(len(s1)+1):
        distance[i][0] = i
    for j in range(len(s2)+1):
        distance[0][j] = j

    # Calculate the distance using the dynamic programming approach
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            distance[i][j] = min(distance[i-1][j] + 1, distance[i][j-1] + 1,
                                 distance[i-1][j-1] + cost)

    # Return the distance
    return distance[len(s1)][len(s2)]