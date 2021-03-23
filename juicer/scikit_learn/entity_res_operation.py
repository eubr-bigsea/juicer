# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from gettext import gettext


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

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.name = 'entity_resolution.IndexingOperation'

        self.has_code = len(self.named_inputs) > 0 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:
            self.attributes = parameters['attributes']
            self.alg = parameters.get(self.ALGORITHM_PARAM, "Block")

            self.input = ""

            self.transpiler_utils.add_import("import recordlinkage as rl")
            if self.alg == "Block":
                self.transpiler_utils.add_import("from recordlinkage.index import Block")

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
            code_columns = code_columns = "\n".join(["indexer.block('{col}')".format(col=col) for col in self.attributes])
            code = """
            indexer = rl.Index()
            {columns_code}
            candidate_links = indexer.index({input})
            {out} = candidate_links.to_frame().reset_index(drop=True)
            {out} = {out}.rename({{0:"DF_1", 1:"DF_2"}}, axis=1)
        """.format(out=self.output,
                   input=self.input,
                   columns_code=code_columns)
            return dedent(code)