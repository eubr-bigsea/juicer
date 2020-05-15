# coding=utf-8
from collections import namedtuple

from pyspark.ml.classification import DecisionTreeClassificationModel

import numpy as np
from graphviz import Digraph, Source
import string
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import random

def choices(population, weights=None, *, cum_weights=None, k=1):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0    # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]

def random_choice():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(choices(alphabet, k=8))

class Node(object):
    def __init__(self, left, right, prediction, impurity_stats, impurity, gain,
                 split, feature, level, threshold, categories=None,
                 features=None):
        if features is None:
            features = {}
        self.impurity = impurity
        self.level = level
        self.feature_index = feature
        self.gain = gain
        self.left = left
        self.right = right
        self.threshold = threshold
        self.prediction = None
        self.direction = None
        self.id = random_choice()
        self.impurity_stats = impurity_stats
        self.categories = categories
        self.features = features

    def __str__(self):
        result = []

        pad = (1 * self.level) * ' '
        result.append(
            '{0}If (feature {1} <= {2})]'.format(pad, self.feature_index,
                                                 self.threshold))
        if self.left is not None:
            result.append('{}'.format(self.left))
        else:
            # import pdb; pdb.set_trace()
            result.append('{0}{0}Predict: {1}'.format(pad, self.prediction))
            # result.append('{}Predict: {}'.format(pad, self.prediction))

        result.append(
            '{}Else If (feature {} > {})'.format(pad, self.feature_index,
                                                 self.threshold))

        if self.right is not None:
            result.append('{}'.format(self.right))
        else:
            result.append('{0}{0}Predict: {1}'.format(pad, self.prediction))
            # result.append('If (feature {} <= {}'.format(
            # self.feature_index, self.threshold))
        return '\n'.join(result)

    def get_label(self):

        if self.threshold is None:
            result = '{0} in {1}\nImpurity: {2} Gain: {3}\nImpurity stats:{4}'
            data = [
                self.features.get(str(self.feature_index), 
                    'Feature {}'.format(self.feature_index)),
                self.categories, round(self.impurity, 4), round(self.gain, 4),
                self.impurity_stats]
        else:
            result = '{0} <= {1}\nImpurity: {2} Gain: {3}\nImpurity stats:{4}'
            data = [
                self.features.get(str(self.feature_index), 
                    'Feature {}'.format(self.feature_index)),
                self.threshold, round(self.impurity, 4), round(self.gain, 4),
                self.impurity_stats]
        return result.format(*data)

    def get_nodes_edges(self, graph):
        green = '#3D9970'
        red = '#FF4136'
        yellow = '#FFDC00'
        blue = '#0074D9'
        navy = '#001f3f'
        orange = '#FF851B'
        purple = '#B10DC9'
        graph.node(self.id, self.get_label())
        if self.left is not None:
            if not isinstance(self.left, LeafNode):
                self.left.get_nodes_edges(graph)
                graph.edge(self.id, self.left.id, label='true', color=orange,
                           fontcolor=orange)
            else:
                graph.node('ResL' + self.id, str(self.left), # peripheries='2',
                           color=green)
                graph.edge(self.id, 'ResL' + self.id, label='true',
                           color=orange, fontcolor=orange)
        else:
            graph.node('L' + self.id,
                       'Predict: {0}'.format(self.prediction))
            graph.edge(self.id, 'L' + self.id, label='true', color='true')

        if self.right is not None:
            if not isinstance(self.right, LeafNode):
                self.right.get_nodes_edges(graph)
                graph.edge(self.id, self.right.id, label='false', color=red,
                           fontcolor=red)
            else:
                graph.node('ResR' + self.id, str(self.right), #peripheries='2',
                           color=green)
                graph.edge(self.id, 'ResR' + self.id, label='false', color=red,
                           fontcolor=red)
        else:
            graph.node('R' + self.id,
                       'Predict: {0}'.format(self.prediction))
            graph.edge(self.id, 'R' + self.id, label='false', color=red,
                       fontcolor=red)


# 
# 
# noinspection PyProtectedMember
def _get_root_node(tree: DecisionTreeClassificationModel):
    if hasattr(tree, 'trees'):
        return tree.trees[0]._call_java('rootNode')
    else:
        return tree._call_java('rootNode')


#
# def get_impurities(tree: DecisionTreeClassificationModel):
#     def recur(node, level, direction):
#         if node.numDescendants() == 0:
#             return None
#         ni = node.impurity()
#         # print(dir(node)) 
#         sp = node.split()
#         nn = No()
#         nn.level = level
#         nn.direction = direction
#         nn.prediction = node.prediction()
#         nn.impurity = node.impurity()
#         nn.feature_index = sp.featureIndex()
#         nn.gain = node.gain()
#         nn.threshold = sp.threshold()
#         nn.left = recur(node.leftChild(), level + 1, 'left')
#         nn.right = recur(node.rightChild(), level + 1, 'rigth')
#         return nn
# 
#     return recur(_get_root_node(tree), 1, 'root')
# 



# https://stackoverflow.com/a/58878355
# LeafNode = namedtuple("LeafNode", ("prediction", "impurity"))


class LeafNode:
    def __init__(self, prediction, impurity):
        self.prediction = prediction
        self.impurity = impurity

    def __str__(self):
        return 'Prediction: {}\nImpurity: {}'.format(self.prediction,
                                                     self.impurity)


InternalNode = namedtuple(
    "InternalNode", (
        "left", "right", "prediction", "impurityStats", 'impurity', 'gain',
        "split"))
CategoricalSplit = namedtuple("CategoricalSplit",
                              ("feature_index", "categories"))
ContinuousSplit = namedtuple("ContinuousSplit", ("feature_index", "threshold"))


def jtree_to_python(jtree, features={}):
    def jsplit_to_python(jsplit):
        if jsplit.getClass().toString().endswith(".ContinuousSplit"):
            return ContinuousSplit(jsplit.featureIndex(), jsplit.threshold())
        else:
            jcat = jsplit.toOld().categories()
            return CategoricalSplit(
                jsplit.featureIndex(),
                [jcat.apply(i) for i in range(jcat.length())])

    def jnode_to_python(jnode, level):
        prediction = jnode.prediction()
        stats = np.array(list(jnode.impurityStats().stats()))
        stats = list(jnode.impurityStats().stats())

        if jnode.numDescendants() != 0:  # InternalNode
            left = jnode_to_python(jnode.leftChild(), level + 1)
            right = jnode_to_python(jnode.rightChild(), level + 1)
            split = jsplit_to_python(jnode.split())
            impurity = jnode.impurity()
            gain = jnode.gain()
            feature = jnode.split().featureIndex()
            cat = None
            try:
                th = jnode.split().threshold()
            except:
                print(dir(jnode.split()))
                th = None
                cat = [x for x in jnode.split().leftCategories()]

            return Node(left, right, prediction, stats, impurity, gain, split,
                        feature, level, th, cat, features)
            # return InternalNode(left, right, prediction, stats,
            # impurity, gain, split)

        else:
            return LeafNode(prediction, stats)

    return jnode_to_python(_get_root_node(jtree), 1)

def get_graph_from_model(m, feat):
    #import pdb;pdb.set_trace()
    tt = jtree_to_python(m, dict([(i, f) for i, f in enumerate(feat)]))
    blue = '#0074D9'
    dot = Digraph()
    dot.attr(splines='polyline')  # , rankdir="LR")
    dot.attr('node', shape='rect', fontname='helvetica', fontsize='7', margin='0')
    color = blue
    dot.attr('node', color=color)
    dot.attr('node', fontcolor='#888888')
    dot.attr('node', fontcolor='#888888')
    dot.attr('edge', splines='polyline', arrowhead='none',
             color=color, fontname='helvetica', fontsize='8', style='dashed')
    tt.get_nodes_edges(dot)

    return Source(dot).pipe(format='svg')


'''
m = DecisionTreeClassificationModel.load('file:///var/tmp/dt4')
# print(get_impurities(m))
# print(m.toDebugString)
tt = jtree_to_python(m,
                     {"0": 'sex', '1': 'pclass', '2': 'embarked2', '3': 'fare'})
# print(tt)

blue = '#0074D9'
dot = Digraph()
dot.attr(splines='polyline')  # , rankdir="LR")
dot.attr('node', shape='rect', fontname='helvetica', fontsize='8')
color = blue
dot.attr('node', color=color)
dot.attr('node', fontcolor='#888888')
dot.attr('node', fontcolor='#888888')
dot.attr('edge', splines='polyline', arrowhead='none',
         color=color, fontname='helvetica', fontsize='8', style='dashed')
tt.get_nodes_edges(dot)

with open('/tmp/lixo.dot', 'w') as f:
    f.write(dot.source)

with open('/tmp/lixo.svg', 'wb') as f:
    f.write(Source(dot).pipe(format='svg'))

'''
