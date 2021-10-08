import json
from textwrap import dedent, indent
from juicer.operation import Operation
from itertools import zip_longest as zip_longest
from gettext import gettext

class MetaPlatformOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task = parameters.get('task')
        self.last = 0

    def get_required_parameter(self, parameters, name):
        if name not in parameters:
            raise ValueError(gettext('Missing required parameter: {}').format(
                name))
        else:
            return parameters.get(name)
    def generate_flow(self):
        return ''
    def set_last(self, value):
        self.last = 1 if value else 0
        return ''
    def _get_task_obj(self):
        return {
            "environment": "DESIGN",
            "id": self.task['id'],
            "forms": {
              "display_schema": {"value": "1"},
              "display_sample": {"value": f"{self.last}"},
              "display_text": {"value": "1"}
            },
            "name": self.task['name'],
            "enabled": True,
        }


class ReadDataOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = True
        self.target_platform = 'scikit-learn'
        self.data_source_id = self.get_required_parameter(parameters, 'data_source')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "mode": {"value": "PERMISSIVE"},
          "data_source": {"value": self.data_source_id},
        })
        return json.dumps(task_obj)
    def xgenerate_code(self):
        return f'''
        {{
        "environment": "DESIGN",
        "id": "{self.task['id']}",
        "forms": {{
          "infer_schema": {{"value": "FROM_LIMONERO"}},
          "mode": {{"value": "PERMISSIVE"}},
          "display_schema": {{"value": "1"}},
          "display_sample": {{"value": "{self.last}"}},
          "data_source": {{"value": {self.data_source_id}}},
          "display_text": {{"value": "1"}}
        }},
        "name": "Read Data",
        "enabled": True,
        "operation": {{"id": 18}}
        }}'''

class TransformOperation(MetaPlatformOperation):
    SLUG_TO_EXPR = {
            'to-upper': 'upper'
        }
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = True
        self.target_platform = 'scikit-learn'
        slug = parameters.get('task').get('operation').get('slug')
        self.attributes = ['Species'] # self.get_required_parameter(parameters, 'attributes')
        x = {
  "value": [
    {
      "alias": "xx",
      "expression": "upper(valor)",
      "error": None,
      "tree": {
        "type": "CallExpression",
        "arguments": [
          {
            "type": "Identifier",
            "name": "valor"
          }
        ],
        "callee": {
          "type": "Identifier",
          "name": "upper"
        }
      }
    },
    {
      "alias": "yyy",
      "expression": "upper(outro)",
      "error": None,
      "tree": {
        "type": "CallExpression",
        "arguments": [
          {
            "type": "Identifier",
            "name": "outro"
          }
        ],
        "callee": {
          "type": "Identifier",
          "name": "upper"
        }
      }
    }
  ],
  "label": "Expressão"
} 

    def generate_code(self):
        return 'Transform Data'

class SortOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = True
        self.target_platform = 'scikit-learn'
        self.attributes = "" # self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        return f'''
        {{
        "environment": "DESIGN",
        "id": "{self.task['id']}",
        "forms": {{
          "attributes": {{"value": [{{"attribute": "Sepal_length", "f": "asc"}}] }},
          "display_schema": {{"value": "1"}},
          "display_sample": {{"value": "{self.last}"}},
          "display_text": {{"value": "1"}}
        }},
        "name": "Sort Data",
        "enabled": True,
        "operation": {{"id": 32}}
        }}'''


