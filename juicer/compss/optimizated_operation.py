# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation


class OptimizatedOperation(Operation):
    """OptimizatedOperation.

    Merge many tasks in only one
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if not 'code_0' in self.parameters and 'code_1' in self.parameters:
            raise ValueError(
                    _("Parameter {} and {} must be informed for task {}")
                    .format('code_0', 'code_1', self.__class__))

        self.code_0 = self.parameters.get('code_0', '')
        self.code_1 = self.parameters.get('code_1', '')
        self.has_code = True  # len(self.named_inputs) == 1
        self.has_code_otm = True
        self.number_tasks = self.parameters.get('number_tasks', 0)
        self.fist_id = self.parameters.get('fist_id', 0)
        self.output = self.named_outputs.get(
                'output data',
                'output_data_{}'.format(self.order))

        print

    def generate_code(self):
        """Generate code."""

        if self.parameters['first_slug'] == 'data-reader':
            self.named_inputs['input data'] = 'hdfs_blocks'

        code = """
        conf = []
        {code}
        {output} = [[] for _ in range(numFrag)]
        for f in range(numFrag):
            {output}[f] = otm_call_{order}({input}[f], conf)
        """.format(code=self.code_0, output=self.output, order=self.order,
                   input=self.named_inputs['input data'])

        return dedent(code)

    def generate_code_otm(self):
        """Generate code."""
        code = ""
        for idx, c in enumerate(self.code_1):
            c = c.replace('conf_X', 'conf_{}'.format(idx))
            code += """conf_{idx} = settings[{idx}]{code}"""\
                .format(idx=idx, code=c)

        return dedent(code)

