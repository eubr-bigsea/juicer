# -*- coding: utf-8 -*-
import ast
import json
import pprint
from textwrap import dedent


from juicer.operation import Operation
from juicer.util import chunks
from juicer.util import dataframe_util
from juicer.util.dataframe_util import get_csv_schema

from juicer.service import limonero_service

class WebServiceInput(Operation):

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = True

        if not self.has_code:
            raise ValueError(
                'input is missing')

    def generate_code(self):
        code = dedent("{output} = 'ws input'".format(output='FIXME'))
        return code


class WebServiceOutput(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = True

        if not self.has_code:
            raise ValueError(
                'Model is being used, but at least one input is missing')

    def generate_code(self):
        code = dedent("{output} = 'ws output".format(output='FIXME'))
        return code


class WebServiceReadModel(Operation):

    ID_WORKFLOW_ID_TASK_ATTRIBUTE = 'id_workflow_id_task'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = True

        if not self.has_code:
            raise ValueError(
                'input is missing')

    def generate_code(self):
        code = dedent("{output} = 'read model'".format(output='FIXME'))
        return code
