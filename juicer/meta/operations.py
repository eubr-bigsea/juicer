import json
from textwrap import dedent, indent
from juicer.operation import Operation
from itertools import zip_longest as zip_longest
from gettext import gettext
from uuid import uuid4

class MetaPlatformOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task = parameters.get('task')
        self.last = 0
        self.new_id = self.task.get('id') + '-0'
        self.output_port_name = 'output data'
        self.input_port_name = 'input data'
        self.has_code = True
        self.target_platform = 'scikit-learn' # FIXME

    def get_required_parameter(self, parameters, name):
        if name not in parameters:
            raise ValueError(gettext('Missing required parameter: {}').format(
                name))
        else:
            return parameters.get(name)

    def generate_flow(self, sibling):

        return json.dumps({
            'source_id': self.new_id, 'target_id': sibling.new_id,
            'source_port_name': self.output_port_name,
            'target_port_name': self.input_port_name,
            'source_port': 0,
            'target_port': 0
        })

    def set_last(self, value):
        self.last = 1 if value else 0
        return ''

    def _get_task_obj(self):
        order = self.task.get('display_order', 0)
        return {
            "id": self.new_id,
            "display_order": self.task.get('display_order', 0),
            "environment": "DESIGN",
            "forms": {
              "display_schema": {"value": "1"},
              "display_sample": {"value": f"{self.last}"},
              "display_text": {"value": "1"}
            },
            "name": self.task['name'],
            "enabled": True,
            "left": (order % 4)* 250 + 100,
            "top": (order // 4)+ 100,
            "z_index": 10
        }


class ReadDataOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.data_source_id = self.get_required_parameter(parameters, 'data_source')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "mode": {"value": "PERMISSIVE"},
          "data_source": {"value": self.data_source_id},
        })
        task_obj['operation'] = {"id": 18}
        return json.dumps(task_obj)

class TransformOperation(MetaPlatformOperation):
    number_re = r'[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    SLUG_TO_EXPR = {
            'extract-numbers': {'f': 'regexp_extract', 'args': [number_re], 'transform': [str]},
            'to-upper': {'f': 'upper'},
            'to-lower': {'f': 'lower'},
            'capitalize': {'f': 'initcap'},
            'remove-accents': {'f': 'strip_accents'},
            'split': {'f': 'split'},
            'trim': {'f': 'trim'},
            'normalize': {'f': 'FIXME'},
            'regexp_extract': {'f': 'regexp_extract', 'args': ['{delimiter}'], 'transform': [str]},
            'round-number': {'f': 'round', 'args': ['{decimals}'], 'transform': [int]},
            'split-into-words': {'f': 'split', 'args': ['{delimiter}'], 'transform': [str]},
            'truncate-text': {'f': 'substring', 'args': ['0', '{characters}'], 'transform': [int, int]},

            'ts-to-date': {'f': 'from_unixtime'},

            'date-to-ts': {'f': 'unix_timestamp'},
            'format-date': {'f': 'date_format', 'args': ['{format}'], 'transform': [str]},
        }
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        op = parameters.get('task').get('operation')
        self.slug = op.get('slug')

        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()
        function_info = self.SLUG_TO_EXPR[self.slug]
        function_name = function_info['f']

        self.form_parameters = {}
        for arg in function_info.get('args', []):
            if arg[0] == '{' and arg[-1] == '}':
                self.form_parameters[arg[1:-1]] = self.parameters.get(arg[1:-1])
        # import sys
        #print(self.form_parameters, file=sys.stderr)

        # Convert the parameters
        function_args = [arg.format(**self.form_parameters) for arg in
            function_info.get('args', [])]

        final_args_str = ''
        final_args = []

        if function_args:
            final_args_str = ', ' + ', '.join(function_args)
            transform = function_info['transform']
            for i, arg in enumerate(function_args):
                v = transform[i](arg)
                final_args.append({'type': 'Literal', 'value': v, 'raw': f'{v}'})
        # Uses the same attribute name as alias, so it will be overwritten
        expressions = []
        for attr in self.attributes:
            expressions.append(
              {
                'alias': attr,
                'expression': f'{function_name}({attr}{final_args_str})',
                'tree': {
                    'type': 'CallExpression',
                    'arguments': [{'type': 'Identifier', 'name': attr}] + final_args,
                    'callee': {'type': 'Identifier', 'name': function_name},
                }
              }
              )
        task_obj['forms']['expression'] = {'value': expressions}
        task_obj['operation'] = {'id': 7}
        return json.dumps(task_obj)

class CleanMissingOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.cleaning_mode = self.get_required_parameter(parameters, 'cleaning_mode')
        self.value = parameters.get('value')
        self.min_missing_ratio = parameters.get('min_missing_ratio')
        self.max_missing_ratio = parameters.get('max_missing_ratio')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['attributes', 'cleaning_mode', 'value', 
            'min_missing_ratio', 'max_missing_ratio']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['operation'] = {"id": 21}
        return json.dumps(task_obj)


class GroupOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.function = parameters.get('function')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          'attributes': {'value': self.attributes},
          'function': {'value': self.function},
        })
        task_obj['operation'] = {"id": 15}
        return json.dumps(task_obj)

class SampleOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.type = parameters.get('type', 'head')
        self.value = parameters.get('value', 50)
        self.seed = parameters.get('seed')
        self.fraction = parameters.get('fraction')
        self.output_port_name = 'sampled data'

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          'type': {'value': self.type},
          'value': {'value': self.value},
          'fraction': {'value': self.fraction},
          'seed': {'value': self.seed},
        })
        task_obj['operation'] = {"id": 28}
        return json.dumps(task_obj)


class FindReplaceOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = True
        self.target_platform = 'scikit-learn'
        self.attributes = "" # self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "attributes": {"value": [{"attribute": "Sepal_length", "f": "asc"}] },
        })
        task_obj['operation'] = {"id": 32}
        return json.dumps(task_obj)

class FilterOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.formula = self.get_required_parameter(parameters, 'formula')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "expression": {"value": self.formula},
        })
        task_obj['operation'] = {"id": 5}
        return json.dumps(task_obj)

class AddByFormulaOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.formula = self.get_required_parameter(parameters, 'formula')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "expression": {"value": self.formula},
        })
        task_obj['operation'] = {"id": 7}
        return json.dumps(task_obj)

class DateDiffOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.alias = self.get_required_parameter(parameters, 'alias')
        self.type = parameters.get('type', 'now')
        self.unit = parameters.get('unit', 'days')
        self.invert = parameters.get('invert', '0') in (1, 'true', True)
        self.date_attribute = parameters.get('date_attribute')
        self.value = parameters.get('value')

    def generate_code(self):
        task_obj = self._get_task_obj()
        formula = {
             'alias': self.alias,
             'expression': f'{function_name}({attr}{final_args_str})',
             'tree': {
                 'type': 'CallExpression',
                 'arguments': [{'type': 'Identifier', 'name': attr}] + final_args,
                 'callee': {'type': 'Identifier', 'name': function_name},
             }
        }
        task_obj['forms'].update({
          "expression": {"value": self.formula},
        })
        task_obj['operation'] = {"id": 7}

        return json.dumps(task_obj)



class SortOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.order_by = self.get_required_parameter(parameters, 'order_by')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "attributes": {"value": self.order_by},
        })
        task_obj['operation'] = {"id": 32}
        return json.dumps(task_obj)
