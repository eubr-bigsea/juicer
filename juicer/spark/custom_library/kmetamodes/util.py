# coding=utf-8
# Author: 'Marissa Saunders' <marissa.saunders@thinkbiganalytics.com>
# License: MIT
# Author: 'Andrey Sapegin' <andrey.sapegin@hpi.de> <andrey@sapegin.org>
# Adapted to Spark Interface by: 'Lucas Miguel Ponce' <lucasmsp@dcc.ufmg.br>

import math


# A method to get maximum value in dict, together with key.
def get_max_value_key(dic):
    v = list(dic.values())
    k = list(dic.keys())
    max_value = max(v)
    key_of_max_value = k[v.index(max_value)]
    return key_of_max_value, max_value


def get_dissim_function(similarity):
    if "hamming" in similarity:
        diss = hamming_dissim
    elif "frequency" in similarity:
        diss = frequency_based_dissim
    else:
        diss = all_frequency_based_dissim_for_modes
    return diss


def hamming_dissim(record, modes):
    """
    Hamming (simple matching) dissimilarity function
    adapted from https://github.com/nicodv/kmodes
    """
    list_dissim = []
    for cluster_mode in modes:
        sum_dissim = 0
        for elem1, elem2 in zip(record, cluster_mode.attrs):
            if elem1 != elem2:
                sum_dissim += 1
        list_dissim.append(sum_dissim)
    return list_dissim


def frequency_based_dissim(record, modes):
    """
    Frequency-based dissimilarity function
    inspired by "Improving K-Modes Algorithm Considering Frequencies of
    Attribute Values in Mode" by He et al.
    """
    list_dissim = []
    for cluster_mode in modes:
        sum_dissim = 0
        for i in range(len(record)):
            if record[i] != cluster_mode.attrs[i]:
                sum_dissim += 1
            else:
                sum_dissim += 1 - cluster_mode.attr_frequencies[i]
        list_dissim.append(sum_dissim)
    return list_dissim


def all_frequency_based_dissim_for_modes(mode, metamodes):
    """
    Andrey Sapegin frequency-based dissimilarity function for clustering of
    modes
    """
    list_dissim = []
    # mode.freq[i] is a set of frequencies for all values of attribute i in
    # the original cluster of this mode metamode.freq[i]
    if mode.freq is None:
        mode.calculate_freq()
    # for each existing cluster metamode
    for metamode in metamodes:
        sum_dissim = 0
        if metamode.freq is None:
            metamode.calculate_freq()
        # for each attribute in the mode
        for i in range(len(mode.attrs)):
            X = mode.freq[i]
            Y = metamode.freq[i]
            # calculate Euclidean dissimilarity between two modes
            sum_dissim += math.sqrt(sum((X.get(d, 0) - Y.get(d, 0)) ** 2
                                        for d in set(X) | set(Y)))
        list_dissim.append(sum_dissim)
    return list_dissim
