# coding=utf-8


from juicer.operation import Operation


class MultiplexerOperation(Operation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.inputs) > 0

    def generate_code(self):
        input_port = self.named_inputs.get('input data 1')
        if input_port is None:
            input_port = self.named_inputs.get('input data 2')
        code = """{out} = {input}""".format(out=self.output, input=input_port)
        return code


class ServiceOutputOperation(Operation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = False

    def generate_code(self):
        pass
