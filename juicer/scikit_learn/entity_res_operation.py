# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from gettext import gettext
import pandas as pd


class IndexingOperation(Operation):
    """
    The indexing module is used to make pairs of records.
    These pairs are called candidate links or candidate matches.
    There are several indexing algorithms available such as blocking and sorted neighborhood indexing.
    Parameters:
    - attributes: list of the attributes that will be used to do de indexing.
    - alg: the algorithm that will ne used for the indexing.
    """

    ATTRIBUTES_PARAM = 'attributes'
    ALGORITHM_PARAM = 'alg'
    WINDOW_PARAM = 'window'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.attributes = parameters['attributes']
            self.alg = parameters.get(self.ALGORITHM_PARAM, "Block")
            self.window = int(parameters.get(self.WINDOW_PARAM, 3))

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            if self.alg == "Block":
                self.transpiler_utils.add_import("from recordlinkage.index import Block")
            elif self.alg == "Full":
                self.transpiler_utils.add_import("from recordlinkage.index import Full")
            elif self.alg == "Random":
                self.transpiler_utils.add_import("from recordlinkage.index import Random")
            elif self.alg == "Sorted Neighbourhood":
                self.transpiler_utils.add_import("from recordlinkage.index import SortedNeighbourhood")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.attributes) == 0:
            raise ValueError(
                _("Parameter '{}' must be x>0 for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        if self.named_inputs.get('input data 1') is not None:
            self.input += self.named_inputs.get('input data 1')
        if self.named_inputs.get('input data 2') is not None:
            if len(self.input) > 0:
                self.input += ","
            self.input += self.named_inputs.get('input data 2')

    def generate_code(self):
        if self.has_code:
            code_columns = None
            if self.alg == "Sorted Neighbourhood":
                code_columns = "\n".join(["indexer.sortedneighbourhood('{col}', window={window})"
                                         .format(col=col,window=self.window) for col in self.attributes])
            elif self.alg == "Block":
                code_columns = "\n".join(["indexer.block('{col}')".format(col=col) for col in self.attributes])
            elif self.alg == "Random":
                code_columns = "\n".join(["indexer.random('{col}')".format(col=col) for col in self.attributes])
            elif self.alg == "Full":
                code_columns = "\n".join(["indexer.full('{col}')".format(col=col) for col in self.attributes])

            code = """
            indexer = rl.Index()
            {columns_code}
            candidate_links = indexer.index({input})
            {out} = candidate_links.to_frame().reset_index(drop=True)
            {out} = {out}.rename({{0:"Record_1", 1:"Record_2"}}, axis=1)
        """.format(out=self.output,
                   input=self.input,
                   columns_code=code_columns)
            return dedent(code)

class ComparingOperation(Operation):
    """
    A set of informative, discriminating and independent features
     is important for a good classification of record pairs into
     matching and distinct pairs.
    The recordlinkage.Compare class and its methods can be used to
     compare records pairs. Several comparison methods are included
     such as string similarity measures, numerical measures and distance measures.
    Parameters:
    - attributes: list of the attributes that will be used to do de comparing.
    """

    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.ComparingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.attributes = parameters['attributes']

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            self.transpiler_utils.add_import("from recordlinkage.compare import Exact")

            self.treatment()

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def treatment(self):
        if len(self.attributes) <= 1:
            raise ValueError(
                _("Parameter '{}' must be x>=2 for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        if self.named_inputs.get('indexing data') is not None:
            self.input = self.named_inputs.get('indexing data')

    def generate_code(self):
        if self.has_code:
            code_columns = None
            code_columns = "\n".join(["compare.exact('{col}', '{col}', label='{col}')".format(col=col) for col in self.attributes])
            code = """
            compare = rl.Compare()
            {columns_code}
            {input} = pd.MultiIndex.from_frame({input}, names=('Record_1', 'Record_2'))
            if input2 is not None and input3 is not None:
                features = compare.compute({input},input2,input3)
            elif input2 is not None:
                features = compare.compute({input},input2)
            elif input3 is not None:
                features = compare.compute({input},input3)
            {out} = features
        """.format(out=self.output,
                   input=self.input,
                   columns_code=code_columns,
                   input2=self.named_inputs.get('input data 2'),
                   input3=self.named_inputs.get('input data 3'))
            return dedent(code)