# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
import re
from juicer.util.template_util import *
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
    - features: list of the attributes that will be used to do de indexing.
    - alg: the algorithm that will ne used for the indexing.
    """

    ATTRIBUTES_PARAM = 'features'
    ALGORITHM_PARAM = 'alg'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'

        self.has_code = any([len(self.named_outputs) > 0, self.contains_results()])
        if self.has_code:
            self.features = parameters['features']
            self.alg = parameters.get(self.ALGORITHM_PARAM, "Block")

        if self.alg = "Block":
            self.transpiler_utils.add_import("from recordlinkage.index import Block")

        self.treatment()

    def treatment(self):
            if len(self.features) == 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 for task {}").format(
                        self.ATTRIBUTES_PARAM, self.__class__))