# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class FeatureAssemblerOperation(Operation):
    """FeatureAssemblerOperation.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti ???????????? / implementation

    Note:   Adicionar um pre valor no alias
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'attributes' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('attributes', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.alias = parameters.get("alias", 'FeatureField')
        self.output = self.named_outputs.get('output data', tmp)

        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.ml.FeatureAssembler " \
                              "import FeatureAssemblerOperation\n"

    def generate_code(self):
        """Generate code."""
        code = """
            cols = {cols}
            alias = '{alias}'
            {output} = FeatureAssemblerOperation({input}, cols, alias, numFrag)
            """.format(output=self.output, alias=self.alias,
                       input=self.named_inputs['input data'],
                       cols=self.parameters['attributes'])

        return dedent(code)


class FeatureIndexerOperation(Operation):
    """FeatureIndexerOperation.

    REVIEW: 2017-10-20
    OK - Juicer / Tahiti ???????????? / implementation

    Note:   Adicionar um pre valor no alias
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if 'attributes' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('attributes', self.__class__))
        elif 'alias' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('alias', self.__class__))

        elif 'type' not in parameters:
            raise ValueError(
                _("Parameters '{}' must be informed for task {}")
                .format('type', self.__class__))

        tmp = 'output_data_{}'.format(self.order)
        self.output = self.named_outputs.get('output data', tmp)
        self.alias = parameters.get("alias", 'FeatureIndexed')
        self.has_code = len(self.named_inputs) == 1
        if self.has_code:
            self.has_import = "from functions.ml.FeatureAssembler " \
                              "import FeatureAssemblerOperation\n"

    def generate_code(self):
        """Generate Code."""
        code = """
            settings = dict()
            settings['inputCol'] = {columns}
            settings['outputCol'] = '{alias}'
            settings['IndexToString'] = False #Currently, only String to Index
            {out}, mapper = FeatureIndexerOperation({input}, settings, numFrag)
            """.format(out=self.output, alias=self.alias,
                       input=self.named_inputs['input data'],
                       columns=self.parameters['attributes'])

        return dedent(code)
