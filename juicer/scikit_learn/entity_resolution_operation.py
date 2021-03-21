# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
import re
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from juicer.util.template_util import *
from juicer.scikit_learn.model_operation import AlgorithmOperation
import json
import time
from random import random
from juicer.spark.expression import Expression


class IndexingOperation(Operation):
    """
    The indexing module is used to make pairs of records.
    These pairs are called candidate links or candidate matches.
    There are several indexing algorithms available such as blocking and sorted neighborhood indexing.
    Parameters:
    - att: list of the attributes that will be used to do de indexing.
    - alg: the algorithm that will ne used for the indexing.
    """

    ATTRIBUTES_PARAM = 'att'
    ALGORITHM_PARAM = 'alg'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'
