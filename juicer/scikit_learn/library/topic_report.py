# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def gen_top_words(model, feature_names, n_top_words):
    """Produces a report for topic identification in text"""
    topics = [i for i in range(len(model.components_))]
    terms = [0] * len(model.components_)
    term_idx = [0] * len(model.components_)
    term_weights = [0] * len(model.components_)

    for topic_idx, topic in enumerate(model.components_):
        terms[topic_idx] = [feature_names[i] for i in
                            topic.argsort()[:-n_top_words - 1:-1]]
        term_idx[topic_idx] = [int(x) for x in
                               topic.argsort()[:-n_top_words - 1:-1]]
        term_weights[topic_idx] = [x for x in
                                   np.sort(topic)[:-n_top_words - 1: -1]]

    df = pd.DataFrame()
    df['{topic_col}'] = topics
    df['{term_idx}'] = term_idx
    df['{terms_weights}'] = term_weights
    df['{terms_col}'] = terms
    return df
