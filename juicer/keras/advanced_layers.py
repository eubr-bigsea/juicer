# -*- coding: utf-8 -*-
from textwrap import dedent

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.util.template_util import *


class PythonCode(Operation):
    CODE_PARAM = 'code'
    OUT_CODE_PARAM = 'out_code'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.output = named_outputs.get('output data',
                                        'out_task_{}'.format(self.order))

        self.code = parameters.get(self.CODE_PARAM, None) or None
        self._out_code = int(parameters.get(self.OUT_CODE_PARAM, 0))

        self.task_name = self.parameters.get('task').get('name')

        if self.CODE_PARAM not in parameters:
            raise ValueError(gettext('Parameter {} is required').format(
                self.CODE_PARAM)
            )

        self.parents_by_port = parameters.get('my_ports', [])
        self.python_code_to_remove = self.remove_python_code_parent()
        self.out_code = False
        self.treatment()

        self.import_code = {'layer': None,
                            'callbacks': [],
                            'model': None,
                            'preprocessing_image': None,
                            'others': None}

        self.has_code = not self.out_code
        self.has_external_python_code_operation = self.out_code

    def remove_python_code_parent(self):
        python_code_to_remove = []
        for parent in self.parents_by_port:
            if parent[0] == 'python code':
                python_code_to_remove.append(convert_parents_to_variable_name(
                    [parent[1]])
                )
        return python_code_to_remove

    def treatment(self):
        self.out_code = True if self._out_code == 1 else False

        if not self.CODE_PARAM:
            raise ValueError(gettext('Parameter {} is required.').format(
                self.CODE_PARAM))

    def generate_code(self):
        return dedent(
            """
            
            # Begin user code - {name}
            
            {code}
            
            # End user code - {name}
            
            """
        ).format(name=self.task_name, code=self.code)
