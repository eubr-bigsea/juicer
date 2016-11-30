from textwrap import dedent

from juicer.spark.operation import Operation


class PearsonCorrelation(Operation):
    """
    Calculates the correlation of two columns of a DataFrame as a double value.
    @deprecated: It should be used as a function in expressions
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        if len(self.inputs) == 1:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = """{} = {}.corr('{}', '{}')""".format(
                output, self.inputs[0], self.attributes[0], self.attributes[1])
        else:
            code = ''

        return dedent(code)