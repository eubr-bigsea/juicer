# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation


class EntityMatchingOperation(Operation):
    ALGORITHM_PARAM = 'algorithm'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.ALGORITHM_PARAM in parameters:
            self.algorithm = parameters[self.ALGORITHM_PARAM]
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ALGORITHM_PARAM, self.__class__))

        self.output = self.named_outputs.get('output data', 'em_data_{}'.format(
            self.order))
        self.has_code = len(self.named_inputs) == 2

    def generate_code(self):
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        code = "{out} = {in1}.unionAll({in2})".format(
            out=self.output, in1=input_data1, in2=input_data2)
        return dedent(code)
