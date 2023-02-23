import json
import pdb
import re
import unicodedata
from collections import namedtuple
from gettext import gettext
from itertools import zip_longest as zip_longest
from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.data_operation import DataReaderOperation
from juicer.spark.etl_operation import AggregationOperation
from juicer.spark.etl_operation import FilterOperation as SparkFilterOperation
from juicer.spark.etl_operation import SampleOrPartitionOperation
from juicer.service import limonero_service

FeatureInfo = namedtuple('FeatureInfo', ['value', 'props', 'type'])

_pythonize_expr = re.compile('[\W_]+')


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def pythonize(s):
    return _pythonize_expr.sub('_', strip_accents(s))


def _as_list(values, transform, size=None):
    # import pdb; pdb.set_trace()
    if values:
        if isinstance(values, list):
            return FeatureInfo(values, None, 'simple_list')
        elif isinstance(values, str):
            return []
        elif (values.get('type') == 'list' or (
                values.get('type') != 'range' and values.get('list'))):
            v = [transform(x) for x in values.get('list', []) if x is not None]
            return FeatureInfo(v, values, 'list')
        elif values.get('type') == 'range':
            qty = size or values.get('quantity') or 3
            _min = values.get('min', 0)
            _max = values.get('max', 3)
            if values.get('distribution') == 'log_uniform':
                return FeatureInfo(
                    f'random.sample(np.log10(np.logspace('
                    f'{_min}, {_max}, {qty})).tolist(), {qty})',
                    values, 'function')
            else:
                return FeatureInfo(
                    f'random.sample(np.linspace('
                    f'{_min}, {_max}, {qty}, dtype=int).tolist(), {qty})',
                    values, 'function')
        elif values.get('type') is None:
            return []
        else:
            return FeatureInfo(
                [transform(x) for x in values if x is not None], None)
    else:
        return []


def _as_boolean_list(values):
    feat = _as_list(values, bool)
    if feat == []:
        return []  # Review how this occurs
    return FeatureInfo(
        [x for x in feat.value if isinstance(x, bool)],
        feat.props, feat.type)


def _as_int_list(values, grid_info):
    size = None
    if 'value' in grid_info:
        value = grid_info.get('value')
        if value.get('strategy') == 'random':
            size = value.get('max_iterations')
    return _as_list(values, int, size)


def _as_float_list(values, grid_info):
    size = None
    if 'value' in grid_info:
        value = grid_info.get('value')
        if value.get('strategy') == 'random':
            size = value.get('max_iterations')

    return _as_list(values, float, size)


def _as_string_list(values):
    return _as_list(values, str)


class MetaPlatformOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task = parameters.get('task')
        self.last = 0
        self.new_id = self.task.get('id') + '-0'
        self.output_port_name = 'output data'
        self.input_port_name = 'input data'
        self.has_code = True
        # self.target_platform = 'scikit-learn'

    def get_required_parameter(self, parameters, name):
        if name not in parameters:
            raise ValueError(gettext('Missing required parameter: {}').format(
                name))
        else:
            return parameters.get(name)

    def generate_flows(self, next_task):
        # import pdb; pdb.set_trace()
        if not self.has_code:
            return ''
        result = [json.dumps({
            'source_id': self.new_id, 'target_id': next_task.new_id,
            'source_port_name': self.output_port_name,
            'target_port_name': next_task.input_port_name,
            'source_port': 0,
            'target_port': 0
        })]
        extra = next_task.generate_extra_flows()
        if extra:
            result.append(extra)

        return ','.join(result)

    def generate_extra_flows(self):
        return None

    def set_last(self, value):
        self.last = 1 if value else 0
        return ''

    def _get_task_obj(self):
        order = self.task.get('display_order', 0)
        wf_type = self.parameters.get('workflow').get('type')

        result = {
            "id": self.new_id,
            "display_order": self.task.get('display_order', 0),
            "environment": "DESIGN",
            "name": self.task['name'],
            "enabled": self.task['enabled'],
            "left": (order % 4) * 250 + 100,
            "top": (order // 4) * 150 + 100,
            "z_index": 10
        }
        if wf_type != 'VIS_BUILDER':
            result.update(
                {'forms': {
                    "display_schema": {"value": "0"},
                    "display_sample": {"value": f"{self.last}"},
                    "sample_size": {"value": self.parameters[
                        'transpiler'].sample_size},
                    "display_text": {"value": "1"}
                }
                })
        else:
            result['forms'] = {}

        return result

    def model_builder_code(self):
        pass

    def visualization_builder_code(self):
        pass


class ReadDataOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.data_source_id = self.get_required_parameter(
            parameters, 'data_source')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "mode": {"value": "PERMISSIVE"},
            "data_source": {"value": self.data_source_id},
        })
        task_obj['operation'] = {"id": 18}
        return json.dumps(task_obj)

    def model_builder_code(self):
        params = {}
        params.update(self.parameters)
        params['mode'] = 'PERMISSIVE'
        dro = DataReaderOperation(params, {}, {'output data': 'df'})
        return dro.generate_code()

    def visualization_builder_code(self):
        return self.model_builder_code()


class TransformOperation(MetaPlatformOperation):
    number_re = r'([\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+)'
    array_index_re = re.compile(r'(?:\D*?)(-?\d+)(?:\D?)')
    SLUG_TO_EXPR = {
        #'extract-numbers': {'f': 'regexp_extract', 'args': [number_re, 1],
        #                    'transform': [str, None]},
        'extract-numbers': {'f': 'extract_numbers', 'args': [],
                            'transform': []},
        'extract-with-regex': {'f': 'regexp_extract', 'args': ['{regex}'],
                               'transform': [str]},
        'replace-with-regex': {'f': 'regexp_replace', 'args': ['{regex}', '{replace}'],
                               'transform': [str, lambda v: '' if v is None else v]},
        'to-upper': {'f': 'upper'},
        'to-lower': {'f': 'lower'},
        'capitalize': {'f': 'initcap'},
        'remove-accents': {'f': 'strip_accents'},
        'parse-to-date': {'f': 'to_date', 'args': ['{format}'], 'transform': [str]},
        'split': {'f': 'split'},
        'trim': {'f': 'trim'},
        'normalize': {'f': 'FIXME'},
        'regexp_extract': {'f': 'regexp_extract', 'args': ['{delimiter}'], 'transform': [str]},
        'round-number': {'f': 'round', 'args': ['{decimals}'], 'transform': [int]},
        'split-into-words': {'f': 'split', 'args': ['{delimiter}'], 'transform': [str]},
        'truncate-text': {'f': 'substring', 'args': ['0', '{characters}'], 'transform': [int, int]},

        'ts-to-date': {'f': 'from_unixtime'},

        'date-to-ts': {'f': 'unix_timestamp'},
        'date-part': {},
        'format-date': {'f': 'date_format', 'args': ['{format}'], 'transform': [str]},
        'truncate-date-to': {'f': 'date_trunc', 'args': ['{format}'], 'transform': [str]},

        'invert-boolean': {'f': None, 'op': '~'},

        'extract-from-array': {'f': None, 'op': ''},
        'concat-array': {'f': 'array_join', 'args': ['{delimiter}'], 'transform': [str]},
        'sort-array': {'f': 'array_sort'},
        'change-array-type': {'f': 'array_cast', 'args': ['{new_type}'], 'transform': [str]},

        'flag-empty': {'f': 'isnull', },
        'flag-with-formula': {'f': None},
    }
    ALIASES = {
        'flag-empty': '_na'
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        op = parameters.get('task').get('operation')
        self.slug = op.get('slug')

        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()

        info = self.SLUG_TO_EXPR[self.slug]

        alias = self.ALIASES.get(self.slug, '')

        function_name = info.get('f')
        expressions = []
        if function_name:

            self.form_parameters = {}
            param_names = []
            for i, (arg, transform) in enumerate(zip(info.get('args', []), info.get(
                    'transform', []))):
                if transform is not None:
                    if arg[0] == '{' and arg[-1] == '}':
                        self.form_parameters[arg[1:-1]] = transform(
                            self.parameters.get(arg[1:-1]))
                        param_names.append(arg[1:-1])
                    else:
                        self.form_parameters[f'param_{i}'] = transform(arg)
                        param_names.append(f'param_{i}')
            # import sys
            #print(self.form_parameters, file=sys.stderr)

            # Convert the parameters
            args = info.get('args', [])
            function_args = [
                (arg.format(**self.form_parameters)
                 if isinstance(arg, str) else str(arg)) for arg in args]

            final_args_str = ''
            final_args = []

            if function_args:
                final_args_str = ', ' + ', '.join(function_args)
                transform = info['transform']
                for i, arg in enumerate(function_args):
                    # if isinstance(args[i], str):
                    #     value = arg
                    # else:
                    #     value = args[i]
                    value = self.form_parameters.get(param_names[i])
                    final_args.append(
                        {'type': 'Literal', 'value': value, 'raw': value})
            # Uses the same attribute name as alias, so it will be overwritten
            for attr in self.attributes:
                expressions.append(
                    {
                        'alias': attr + alias,
                        'expression': f'{function_name}({attr}{final_args_str})',
                        'tree': {
                            'type': 'CallExpression',
                            'arguments': [{'type': 'Identifier', 'name': attr}] + final_args,
                            'callee': {'type': 'Identifier', 'name': function_name},
                        }
                    })
        elif self.slug == 'invert-boolean':
            for attr in self.attributes:
                expressions.append(
                    {
                        'alias': attr,
                        'expression': f'!{attr}',
                        'tree': {
                            'type': 'UnaryExpression', 'operator': '~',
                            'argument': {'type': 'Identifier',  'name': attr},
                            'prefix': True
                        }
                    })
        elif self.slug == 'date-add':
            source_type = self.parameters.get('type', 'constant')
            if source_type == 'constant':
                source = self.parameters.get('value', 0)
            else:
                source = self.parameters.get('value_attribute').get(0)

            component_to_function = {
                'second': 'seconds',
                'minute': 'minutes',
                'hour': 'hours',
                'day': 'days',
                'week': 'weeks',
                'month': 'months',
                'year': 'years',
            }
            component = self.parameters.get('component', 'day')
            f = component_to_function.get(component)
            for attr in self.attributes:
                expressions.append(
                    {
                        'alias': f'{attr}_{f}',
                        'expression': f'{f}("{attr}")',
                        'tree': {
                            'type': 'CallExpression',
                            'arguments': [
                                {'type': 'Identifier',  'name': attr}
                            ],
                            'callee': {'type': 'Identifier', 'name': f},
                        }
                    })

        elif self.slug == 'date-part':
            component = self.parameters.get('component', 'day')
            component_to_function = {
                'second': 'second',
                'minute': 'minute',
                'hour': 'hour',
                'day': 'dayofmonth',
                'week': 'weekofyear',
                'month': 'month',
                'year': 'year',
            }
            f = component_to_function.get(component)
            for attr in self.attributes:
                expressions.append(
                    {
                        'alias': f'{attr}_{f}',
                        'expression': f'{f}("{attr}")',
                        'tree': {
                            'type': 'CallExpression',
                            'arguments': [
                                {'type': 'Identifier',  'name': attr}
                            ],
                            'callee': {'type': 'Identifier', 'name': f},
                        }
                    })
        elif self.slug == 'extract-from-array':
            indexes = [int(x) for x in self.array_index_re.findall(
                self.parameters.get('indexes', '0') or '0')] or [0]
            attr = self.attributes[0]
            for index in indexes:
                suffix = f'{index}' if index > -1 else f'n{-1*index}'
                expressions.append(
                    {
                        'alias': f'{attr}_{suffix}',
                        'expression': f'element_at("{attr}", {index})',
                        'tree': {
                            'type': 'CallExpression',
                            'arguments': [
                                {'type': 'Identifier',  'name': attr},
                                {'type': 'Literal',  'value': index,
                                    'raw': f'"{index}"'}
                            ],
                            'callee': {'type': 'Identifier', 'name': 'element_at'},
                        }
                    })

        elif self.slug == 'flag-with-formula':
            expressions.extend(self.parameters.get('formula'))

        task_obj['forms']['expression'] = {'value': expressions}
        task_obj['operation'] = {'id': 7}
        return json.dumps(task_obj)


class CleanMissingOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.cleaning_mode = self.get_required_parameter(
            parameters, 'cleaning_mode')
        self.value = parameters.get('value')
        self.min_missing_ratio = parameters.get('min_missing_ratio')
        self.max_missing_ratio = parameters.get('max_missing_ratio')
        self.output_port_name = 'output result'

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['attributes', 'cleaning_mode', 'value',
                     'min_missing_ratio', 'max_missing_ratio']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['operation'] = {"id": 21}
        return json.dumps(task_obj)


class CastOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.cast_attributes = self.get_required_parameter(
            parameters, 'cast_attributes')
        self.errors = self.get_required_parameter(parameters, 'errors')
        self.invalid_values = parameters.get('invalid_values')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['cast_attributes', 'errors', 'invalid_values']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['operation'] = {"id": 140}
        return json.dumps(task_obj)


class StringIndexerOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['attributes']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['forms']['alias'] = {'value': ','.join(
            [f'{a}_inx' for a in self.attributes])}
        task_obj['operation'] = {"id": 40}
        return json.dumps(task_obj)


class OneHotEncodingOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['attributes']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['forms']['alias'] = {'value': ','.join(
            [f'{a}_ohe' for a in self.attributes])}
        task_obj['operation'] = {"id": 75}
        return json.dumps(task_obj)


class GroupOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = parameters.get('attributes')
        self.function = parameters.get('function')

        for f in self.function:
            if f['f'] == '':
                f['f'] = 'ident'
            else:
                f['f'] = f['f'].lower()

        self.has_code = bool(self.attributes)

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            'attributes': {'value': self.attributes},
            'function': {'value': self.function},
        })
        task_obj['operation'] = {"id": 15}
        return json.dumps(task_obj)

    def visualization_builder_code(self):
        params = {}
        params.update(self.parameters)
        agg = AggregationOperation(params, {}, {'input data': 'df',
                                                'output data': 'df'})
        return agg.generate_code()


class SampleOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.type = parameters.get('type')
        self.value = int(parameters.get('value', 0))
        self.seed = parameters.get('seed')
        self.fraction = parameters.get('fraction')
        self.output_port_name = 'sampled data'

        self.has_code = self.value > 0

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

    def model_builder_code(self):
        spo = SampleOrPartitionOperation(self.parameters, {'input data': 'df'},
                                         {'sampled data': 'df'})
        return spo.generate_code()

    def visualization_builder_code(self):
        params = {}
        params.update(self.parameters)
        params['type'] = 'value'
        params['value'] = self.parameters.get('value', 50)
        dro = SampleOrPartitionOperation(params, {'input data': 'df'},
                                         {'sampled data': 'df'})
        return dro.generate_code()


class RescaleOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.type = parameters.get('type')
        self.attributes = parameters.get('attributes')

        if self.type == 'min_max':
            self.operation_id = 91
            self.forms = {
                'min': {'value': float(parameters.get('min', 0))},
                'max': {'value': float(parameters.get('max', 1))},
            }
        elif self.type == 'z_score':
            self.operation_id = 90
            self.forms = {
                'with_std': {'value': parameters.get('with_std')},
                'with_mean': {'value': parameters.get('with_mean')}
            }
        else:
            self.operation_id = 92
            self.forms = {}

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update(self.forms)
        task_obj['forms'].update({
            'attributes': {'value': self.attributes},
        })
        task_obj['operation'] = {"id": self.operation_id}
        return json.dumps(task_obj)


class ForceRangeOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.start = float(self.get_required_parameter(parameters, 'start'))
        self.end = float(self.get_required_parameter(parameters, 'end'))

        if self.start % 1 == 0:
            self.start = int(self.start)
        if self.end % 1 == 0:
            self.end = int(self.end)
        self.outliers = parameters.get('outliers', 'clip')
        self.operation_id = 7

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'] = {}
        task_obj['forms'].update({
            'attributes': {'value': self.attributes},
        })
        task_obj = self._get_task_obj()
        task_obj['operation'] = {"id": self.operation_id}

        expressions = []
        if self.outliers == 'clean':
            for attr in self.attributes:
                expression = (
                    f'when(({attr} < {self.start}) | ({attr} > {self.end}), None, {attr})')
                formula = {
                    'alias': attr,
                    'expression': expression,
                    'tree': {
                        "type":"CallExpression",
                        "arguments":[
                            {
                                "type":"BinaryExpression",
                                "operator": "|",
                                "left":{
                                    "type":"BinaryExpression",
                                    "operator":"<",
                                    "left":{
                                        "type":"Identifier",
                                        "name": attr
                                     },
                                     "right":{
                                        "type":"Literal",
                                        "value": self.start,
                                        "raw": self.start
                                     }
                                },
                                "right":{
                                    "type":"BinaryExpression",
                                    "operator":">",
                                    "left":{
                                        "type":"Identifier",
                                        "name": attr
                                    },
                                    "right":{
                                        "type":"Literal",
                                        "value": self.end,
                                        "raw": self.end
                                    }
                                }
                            },
                            {
                                "type":"Literal",
                                "value": None,
                                "raw": None
                            },
                            {
                                "type": "Identifier",
                                "name": attr
                            }
                         ],
                         "callee":{"type":"Identifier","name":"when"}}
                    }
                expressions.append(formula)
        else:
            for attr in self.attributes:
                expression = (
                    f'when({attr} < {self.start}, {self.start}, {attr} > {self.end}, self.end, {attr})')
                formula = {
                    'alias': attr,
                    'expression': expression,
                    'tree': {
                        "type":"CallExpression",
                        "arguments":[
                            {
                                "type":"BinaryExpression",
                                "operator":"<",
                                "left":{
                                   "type":"Identifier",
                                   "name": attr
                                },
                                "right":{
                                   "type":"Literal",
                                   "value": self.start,
                                   "raw": self.start
                                }
                            },
                            {
                                "type": "Literal",
                                "value": self.start,
                                "raw": self.start,
                            },
                            {
                                "type":"BinaryExpression",
                                "operator":">",
                                "left":{
                                    "type":"Identifier",
                                    "name": attr
                                },
                                "right":{
                                    "type":"Literal",
                                    "value": self.end,
                                    "raw": self.end
                                }
                            },
                            {
                                "type":"Literal",
                                "value": self.end,
                                "raw": self.end
                            },
                            {
                                "type": "Identifier",
                                "name": attr
                            }
                         ],
                         "callee":{"type":"Identifier","name":"when"}}
                    }
                expressions.append(formula)
        task_obj['forms'].update({
            "expression": {"value": expressions},
        })
        return json.dumps(task_obj)

class FindReplaceOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.has_code = True

        self.attributes = self.get_required_parameter(parameters, 'attributes')

        self.forms = {'value': {'value': parameters.get('find')}}
        self.nullify = parameters.get('nullify') in ('1', 1)
        if self.nullify:
            self.forms['replacement'] = {'value': None}
        else: 
            self.forms['replacement'] = {'value': parameters.get('replace')}
        self.forms['nullify'] = {'value': self.nullify}
        self.operation_id = 27

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update(self.forms)
        task_obj['forms'].update({
            'attributes': {'value': self.attributes},
        })
        task_obj['operation'] = {"id": self.operation_id}
        return json.dumps(task_obj)

        task_obj = self._get_task_obj()
        attr = self.attributes
        is_number = True
        replacement = self.replace
        if not self.nullify:
            try:
                find = float(self.find)
                float(self.replace)
            except:
                is_number = False
        else:
            replacement = None
            is_number = False

        if is_number:
            expression = (f'when({attr} == "{self.find}", {self.replace}", '
                          f'{attr} == {self.find}, {self.replace}, {attr})')
        else:
            expression = f'when({attr} == {self.find}, {replacement}, {attr})'
        if not is_number:
            formula = {
                'alias': attr,
                'expression': expression,
                'tree': {
                    'type': 'CallExpression',
                    'arguments': [
                        {'type': 'BinaryExpression', 'operator': '==',
                         'left': {'type': 'Identifier', 'name': attr},
                         'right': {'type': 'Literal', 'value': self.find,
                                   'raw': f'{self.find}'}
                         },
                        {'type': 'Literal', 'value': replacement,
                            'raw': f'{replacement}'},
                        {'type': 'Identifier', 'name': attr},
                    ],
                    'callee': {'type': 'Identifier', 'name': 'when'},
                }
            }
        else:
            find = float(self.find) if '.' in self.find else int(self.find)
            replace = float(self.replace) if '.' in self.replace else int(
                self.replace)
            formula = {
                'alias': attr,
                'expression': expression,
                'tree': {
                    'type': 'CallExpression',
                    'arguments': [
                        {'type': 'BinaryExpression', 'operator': '==',
                         'left': {'type': 'Identifier', 'name': attr},
                         'right': {'type': 'Literal', 'value': find,
                                   'raw': find}
                         },
                        {'type': 'Literal', 'value': replace,
                            'raw': replace},
                        {'type': 'Identifier', 'name': attr},
                    ],
                    'callee': {'type': 'Identifier', 'name': 'when'},
                }
            }
        task_obj['forms'].update({
            "expression": {"value": [formula]},
        })
        task_obj['operation'] = {"id": 7}
        return json.dumps(task_obj)


class FilterOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.formula = self.get_required_parameter(parameters, 'formula')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "expression": {"value": self.formula},
        })
        task_obj['operation'] = {"id": 5}
        return json.dumps(task_obj)

    def visualization_builder_code(self):
        params = {}
        params.update(self.parameters)
        flter = SparkFilterOperation(params, {}, {'input data': 'df',
                                                  'output data': 'df'})
        return flter.generate_code()


class RemoveMissingOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def _get_tree(self, attributes):
        if len(attributes) > 1:
            return {
                'type': 'LogicalExpression',
                'operator': '&&',
                'left': self._get_tree(attributes[1:]),
                'right': {
                    'type': 'CallExpression',
                    'arguments': [
                        {'type': 'Identifier', 'name': attributes[0]}
                    ],
                    'callee': {'type': 'Identifier', 'name': 'isnotnull'}
                }
            }
        elif len(attributes) == 1:
            return {
                'type': 'CallExpression',
                'arguments': [
                        {'type': 'Identifier', 'name': attributes[0]}
                ],
                'callee': {'type': 'Identifier', 'name': 'isnotnull'}
            }

    def generate_code(self):
        task_obj = self._get_task_obj()
        conditions = [f'isnotnull({attr})' for attr in self.attributes]
        expression = [
            {
                "alias": "filter",
                "expression": ' && '.join(conditions),
                "tree": self._get_tree(self.attributes)
            }
        ]
        task_obj['forms'].update({"expression": {"value": expression}})
        task_obj['operation'] = {"id": 5}
        return json.dumps(task_obj)


class AddByFormulaOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
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
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
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
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.order_by = self.get_required_parameter(parameters, 'order_by')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "attributes": {"value": self.order_by},
        })
        task_obj['operation'] = {"id": 32}
        return json.dumps(task_obj)


class SelectOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.mode = parameters.get('mode', 'include') or 'include'
        self.output_port_name = 'output projected data'

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "attributes": {"value": self.attributes},
            "mode": {"value": self.mode},
        })
        task_obj['operation'] = {"id": 6, 'slug': 'projection'}
        return json.dumps(task_obj)


class RenameOperation(SelectOperation):
    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "attributes": {"value": self.attributes},
            "mode": {"value": "rename"},
        })
        task_obj['operation'] = {"id": 6}
        return json.dumps(task_obj)


class DiscardOperation(SelectOperation):
    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "attributes": {"value":
                           [{'attribute': a} for a in self.attributes]},
            "mode": {"value": "exclude"},
        })
        task_obj['operation'] = {"id": 6}
        return json.dumps(task_obj)


class DuplicateOperation(SelectOperation):
    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "attributes": {"value": self.attributes},
            "mode": {"value": "duplicate"},
        })
        task_obj['operation'] = {"id": 6}
        return json.dumps(task_obj)


class ExtractFromArrayOperation(MetaPlatformOperation):
    exp_index = re.compile(r'\b(\d+)\b')

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.indexes = self.exp_index.findall(
            parameters.get('indexes', '') or '')


class SaveOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.parameter_names = ['name', 'path', 'format', 'tags',
                                'mode', 'header', 'storage']

    def generate_code(self):
        task_obj = self._get_task_obj()
        for param in self.parameter_names:
            task_obj['forms'][param] = {'value': self.parameters.get(param)}
        task_obj['operation'] = {"id": 30}
        return json.dumps(task_obj)


class ConcatRowsOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.data_source = self.get_required_parameter(
            parameters, 'data_source')
        self.task_id = self.task.get('id')
        self.other_id = f'{self.task_id}-1'
        self.input_port_name = 'input data 1'

    def generate_code(self):
        order = self.task.get('display_order', 0)
        task_obj = self._get_task_obj()
        task_obj['operation'] = {"id": 12}

        other_task = {
            "id": self.other_id,
            "display_order": 100,
            "environment": "DESIGN",
            "forms": {
                "display_schema": {"value": "0"},
                "display_sample": {"value": "0"},
                "display_text": {"value": "0"},
                "data_source": {"value": self.data_source},
            },
            "name": gettext('Read data'),
            "enabled": True,
            "left": (order % 4) * 250 + 100,
            "top": (order // 4) * 150 + 175,
            "z_index": 100,
            "operation": {"id": 18}
        }

        return json.dumps(task_obj) + ',' + json.dumps(other_task)

    def generate_extra_flows(self):
        return json.dumps({
            'source_id': self.other_id,
            'target_id': f'{self.task_id}-0',
            'source_port_name': 'output data',
            'target_port_name': 'input data 2',
            'source_port': 0,
            'target_port': 0
        })


class JoinOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        self.data_source = self.get_required_parameter(
            parameters, 'data_source')

        self.task_id = self.task.get('id')
        self.other_id = f'{self.task_id}-1'

        self.input_port_name = 'input data 1'
        self.parameter_names = ['keep_right_keys',
                                'match_case', 'join_parameters']

    def generate_code(self):
        order = self.task.get('display_order', 0)
        task_obj = self._get_task_obj()
        task_obj['operation'] = {"id": 16}

        for param in self.parameter_names:
            task_obj['forms'][param] = {'value': self.parameters.get(param)}

        other_task = {
            "id": self.other_id,
            "display_order": 100,
            "environment": "DESIGN",
            "forms": {
                "display_schema": {"value": "0"},
                "display_sample": {"value": "0"},
                "display_text": {"value": "0"},
                "data_source": {"value": self.data_source},
            },
            "name": gettext('Read data'),
            "enabled": True,
            "left": (order % 4) * 250 + 100,
            "top": (order // 4) * 150 + 175,
            "z_index": 100,
            "operation": {"id": 18}
        }

        return json.dumps(task_obj) + ',' + json.dumps(other_task)

    def generate_extra_flows(self):
        return json.dumps({
            'source_id': self.other_id,
            'target_id': f'{self.task_id}-0',
            'source_port_name': 'output data',
            'target_port_name': 'input data 2',
            'source_port': 0,
            'target_port': 0
        })


class GenerateNGramsOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        self.attributes = self.get_required_parameter(parameters, 'attributes')
        # self.mode = self.get_required_parameter(parameters, 'mode')
        self.n = self.get_required_parameter(parameters, 'n')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['attributes', 'n']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['operation'] = {"id": 51}
        return json.dumps(task_obj)


class ModelMetaOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task = parameters.get('task')

        self.has_code = True


class EstimatorMetaOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs, task_type):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = "CHANGE_VAR"
        self.name = "CHANGE_NAME"
        self.hyperparameters = {}
        self.task_type = task_type
        self.grid_info = parameters.get('workflow').get(
            'forms', {}).get('$grid', {})

    def get_constrained_params(self):
        return None

    def generate_code(self):
        code = []
        name = self.name
        var = self.var
        variations = self.get_variations()

        if len(variations) > 0:
            for i, (class_name, params) in enumerate(variations):
                code.append(dedent(f"""
                    {var}_{i} = {self.task_type}.{class_name}(**alg_params)
        
                    # Lemonade internal use
                    {var}_{i}.task_id = '{self.task.get('id')}'
                    {var}_{i}.task_name = '{self.task.get('name')} ({class_name})'
                    {var}_{i}.operation_id = '{self.task.get('operation').get('id')}'
                    """))
        else:
            code.append(dedent(f"""
                {var} = {self.task_type}.{name}(**alg_params)
    
                # Lemonade internal use
                {var}.task_id = '{self.task.get('id')}'
                {var}.task_name = '{self.task.get('name')}'
                {var}.operation_id = '{self.task.get('operation').get('id')}'
                """))
        return '\n'.join(code).strip()

    def generate_hyperparameters_code(self, var=None, index=None, invalid=None):
        code = []
        if var == None:
            var = self.var
        else:
            var = f'{var}_{index}'
        if invalid is None:
            invalid = []
        grid_strategy = self.grid_info.get('value', {}).get('strategy', 'grid')
        for name, value in self.hyperparameters.items():
            if value is not None and value and len(value.value) > 0:
                if name not in invalid:
                    if grid_strategy == 'grid' or value.type == 'simple_list':
                        code.append(
                            f'.addGrid({var}.{name}, {self.parse(value)})')

        code.append('.build()')
        return '\\\n'.join(code)

    def generate_random_hyperparameters_code(self):
        code = []
        grid_strategy = self.grid_info.get('value', {}).get('strategy', 'grid')
        random_hyperparams = []
        for name, value in self.hyperparameters.items():
            if value is not None and value and len(value.value) > 0:
                if grid_strategy != 'grid' and value.type != 'simple_list':
                    random_hyperparams.append((name, value))

        if random_hyperparams:
            code.append(
                '# Random hyperparameters and/or parameters with restrictions')
            if len(random_hyperparams) == 1:
                name, value = random_hyperparams[0]
                code.append(
                    f'rnd_params = [[({self.var}.{name}, v) for v in {value.value}]] # {name}')
            else:
                code.append('rnd_params = zip(')
                for name, value in random_hyperparams:
                    code.append(
                        f'    [({self.var}.{name}, v) for v in {value.value}], # {name}')
                code.append(')')
            code.append('\n# Cartesian product between ')
            code.append('# grid and random hyper-parameters')
            code.append(
                f'tmp_{self.var} = list(itertools.product(grid_{self.var}, rnd_params))')
            code.append(dedent(f"""
                grid_{self.var} = []
                for grid_p, rand_p in tmp_{self.var}:
                    grid_{self.var}.append({{**grid_p, **dict(rand_p)}})
                    
            """).strip())
        constrained = self.get_constrained_params()
        if constrained:
            code.append('# Constrained params')
            code.append(
                '# i.e. GeneralizedLinearRegression with family and link')
            code.append('constrained_params = [')

            for constrained in self.get_constrained_params():
                code.append(f'    {constrained},')
            code.append(f']')

            code.append('\n# Cartesian product between ')
            code.append('# grid and constrained hyper-parameters')
            code.append(
                f'tmp_{self.var} = list(itertools.product(grid_{self.var}, constrained_params))')
            code.append(dedent(f"""
                grid_{self.var} = []
                for grid_p, const_p in tmp_{self.var}:
                    grid_{self.var}.append({{**grid_p, **dict(const_p)}})"""))

        return '\n'.join(code)

    def parse(self, value):
        if isinstance(value, list):
            return repr([x.value for x in value if x is not None])
        elif isinstance(value, FeatureInfo) and value.type == 'function':
            return value.value
        return repr(value.value)

    def get_variations(self):
        return []


class EvaluatorOperation(ModelMetaOperation):
    TYPE_TO_CLASS = {
        'binary-classification': 'BinaryClassificationEvaluator',
        'multiclass-classification': 'MulticlassClassificationEvaluator',
        'regression': 'RegressionEvaluator',
        'clustering': 'ClusteringEvaluator',
    }
    TYPE_TO_METRIC_PARAM = {
        'binary-classification': 'bin_metric',
        'multiclass-classification': 'multi_metric',
        'regression': 'reg_metric',
        'clustering': 'clust_metric',
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.task_type = parameters.get('task_type', 'binary-classification')
        self.task_name = parameters.get('task').get('name')
        self.metric = parameters.get(self.TYPE_TO_METRIC_PARAM[self.task_type])
        self.task_id = parameters.get('task').get('id')
        self.operation_id = parameters.get('task').get('operation').get('id')

    def generate_code(self):
        evaluator = self.TYPE_TO_CLASS[self.task_type]
        meta_form = self.parameters.get('workflow').get(
            'forms', {}).get('$meta', {}).get('value')
        if meta_form.get('taskType') != 'clustering':
            label = 'labelCol=label'
        else:
            label = ''  # clustering doesn't support it

        code = dedent(
            f"""
            evaluator = evaluation.{evaluator}(metricName='{self.metric}', {label})
            evaluator.task_id = '{self.task_id}'
            evaluator.operation_id = {self.operation_id}
            """)
        return code.strip()


class FeaturesOperation(ModelMetaOperation):
    TRANSFORM_TO_SUFFIX = {
        'binarize': 'bin',
        'not_null': 'not_null',
        'quantis': 'quantis',
        'buckets': 'buckets'
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        features = parameters.get('features')
        self.all_attributes = [f for f in features if f.get('enabled')]
        self.features = []
        self.numerical_features = []
        self.categorical_features = []
        self.textual_features = []
        self.features_names = []

        for f in features:
            if f.get('usage') == 'unused':
                continue
            f['var'] = pythonize(f['name'])
            transform = f.get('transform')
            name = f['name']
            self._separate_label_from_feature(f)

            missing = f.get('missing_data')
            if missing:
                f['na_name'] = f'{name}_na'
            else:
                f['na_name'] = name

            # Feature used as is
            final_name = name

            f['final_name'] = final_name

        meta_form = self.parameters.get('workflow').get(
            'forms', {}).get('$meta', {}).get('value')
        if meta_form.get('taskType') != 'clustering' and self.label is None:
            raise ValueError(gettext(
                'Missing required parameter: {}').format('label'))
        if len(self.features) == 0:
            raise ValueError(gettext(
                'Missing required parameter: {}').format('features'))

    def _separate_label_from_feature(self, f):
        if f['usage'] == 'label':
            self.label = f
        elif f['usage'] == 'feature':
            self.features.append(f)
            if f['feature_type'] == 'numerical':
                self.numerical_features.append(f)
            elif f['feature_type'] == 'categorical':
                self.categorical_features.append(f)
            elif f['feature_type'] == 'textual':
                self.textual_features.append(f)

    def generate_code(self):
        code = []
        for f in self.features:
            name = f.get('name')
            transform = f.get('transform')
            data_type = f.get('feature_type')
            missing = f.get('missing_data')
            scaler = f.get('scale')

            if data_type == 'numerical':
                if transform in ('keep', '', None):
                    final_name = name
                elif transform == 'binarize':
                    final_name = name + '_binz'
                    threshold = self.parameters.get('threshold', 0.0)
                    code.append(dedent(f"""
                        {f['var']}_bin = feature.Binarizer(
                            threshold={threshold}, inputCol='{f['na_name']}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_bin) """))
                elif transform in ('quantiles', 'quantis'):
                    final_name = name + '_qtles'
                    num_buckets = f.get('quantis', 2)
                    code.append(dedent(f"""
                        {f['var']}_qtles = feature.QuantileDiscretizer(
                            numBuckets={num_buckets}, inputCol='{f['na_name']}',
                            outputCol='{final_name}', handleInvalid='skip')
                        features_stages.append({f['var']}_qtles) """))
                elif transform == 'buckets':
                    splits = ', '.join([str(x) for x in sorted(
                        [float(x) for x in f.get('buckets')])])
                    if splits:
                        final_name = name + '_bkt'
                        code.append(dedent(f"""
                            {f['var']}_qtles = feature.Bucketizer(
                                splits=[-float('inf'), {splits}, float('inf')],
                                inputCol='{f['na_name']}',
                                outputCol='{final_name}', handleInvalid='skip')
                            features_stages.append({f['var']}_qtles) """))
                    else:
                        final_name = None
                if scaler:
                    old_final_name = final_name
                    final_name = name + '_scl'
                    if scaler == 'min_max':
                        scaler_cls = 'MinMaxScaler'
                    elif scaler == 'standard':
                        scaler_cls = 'StandardScaler'
                    else:
                        scaler_cls = 'MaxAbsScaler'
                    code.append(dedent(f"""
                        {f['var']}_asm = feature.VectorAssembler(
                            handleInvalid='skip',
                            inputCols=['{old_final_name}'],
                            outputCol='{f['var']}_asm')
                        features_stages.append({f['var']}_asm)
                        {f['var']}_scl = feature.{scaler_cls}(
                            inputCol='{f['var']}_asm',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_scl) """))

                if final_name is not None:
                    self.features_names.append(final_name)
            elif data_type == 'categorical':
                if missing == 'constant' and transform != 'not_null':
                    cte = f.get('constant')
                    stmt = f"SELECT *, COALESCE({f['name']}, '{cte}') AS {f['var']}_na FROM __THIS__"
                    code.append(dedent(f"""
                        {f['var']}_na = feature.SQLTransformer(
                            statement="{stmt}")
                        features_stages.append({f['var']}_na) """))
                elif missing == 'remove' and transform != 'not_null':
                    stmt = f"SELECT * FROM __THIS__ WHERE NOT ISNULL({f['name']})"
                    code.append(dedent(f"""
                        {f['var']} = feature.SQLTransformer(
                            statement="{stmt}")
                        features_stages.append({f['var']}) """))
                    f['na_name'] = f['name']

                if transform == 'not_null':
                    final_name = name + '_na'
                    stmt = f"SELECT *, INT(ISNULL({f['na_name']})) AS {f['var']}_na FROM __THIS__"
                    code.append(dedent(f"""
                        {f['var']}_na = feature.SQLTransformer(
                            statement='{stmt}')
                        features_stages.append({f['var']}_na) """))

                else:  # transform in ('string_indexer', '', None):
                    final_name = name + '_inx'
                    code.append(dedent(f"""
                        {f['var']}_inx = feature.StringIndexer(
                            inputCol='{f['na_name']}',
                            outputCol='{final_name}',
                            handleInvalid='skip')
                        features_stages.append({f['var']}_inx) """))

                if transform == 'one_hot_encoder':
                    old_final_name = final_name
                    final_name = name + '_ohe'
                    code.append(dedent(f"""
                        {f['var']}_ohe = feature.OneHotEncoder(
                            inputCol='{old_final_name}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_ohe) """))

                self.features_names.append(final_name)

        return '\n'.join(code).strip()

    def get_final_features_names(self):
        return self.features_names

    def generate_code_for_missing_data_handling(self):
        code = []

        by_constant = []
        by_media = []
        by_median = []
        to_remove = []
        for f in self.all_attributes:
            na = f.get('missing_data')
            if na == 'media':
                by_media.append(f)
            elif na == 'median':
                by_median.append(f)
            elif na == 'constant':
                if f['feature_type'] == 'numerical':
                    # Adjust single quote for texts
                    f['constant'] = (float(f['constant'])
                                     if '.' in f['constant']
                                     else int(f['constant']))
                by_constant.append(f)
            elif na == 'remove':
                to_remove.append(f)

        if by_constant:
            replacements = dict([(f['name'], f['constant'])
                                 for f in by_constant])
            code.append(
                f'df = df.na.fill({json.dumps(replacements, indent=4)})')
        if to_remove:
            subset = [f['name'] for f in to_remove]
            code.append(
                f'df = df.dropna(subset={json.dumps(subset, indent=4)})')

        def cast_to_double(code_list, attributes):
            code_list.append(
                '# Spark 2.x Imputer only supports double and float')
            for attr in attributes:
                code_list.append(
                    f"df = df.withColumn('{attr}', df['{attr}'].cast('double'))")
        if by_media:
            subset = [f['name'] for f in by_media]
            # cast_to_double(code, subset)
            code.append(dedent(
                f"""
                    imputer = feature.Imputer(
                        strategy='mean',
                        inputCols={json.dumps(subset)},
                        outputCols={json.dumps(subset)})
                    df = imputer.fit(df).transform(df)
                """))
        if by_median:
            subset = [f['name'] for f in by_median]
            # cast_to_double(code, subset)
            code.append(dedent(
                f"""
                    imputer = feature.Imputer(
                        strategy='median',
                        inputCols={json.dumps(subset)},
                        outputCols={json.dumps(subset)})
                    df = imputer.fit(df).transform(df)
                """))

        if code:
            return '\n'.join([c.strip() for c in code])
        else:
            return '# nothing to handle'


class FeaturesReductionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.method = parameters.get('method', 'pca') or 'pca'
        self.k = int(parameters.get('k', 2) or 2)
        if self.method == 'disabled':
            self.has_code = False

    def generate_code(self):
        code = dedent(f"""
            pca_va = feature.VectorAssembler(
                inputCols=numerical_features, outputCol='pca_input_features',
                handleInvalid='skip')
            feature_reducer = feature.PCA(k={self.k},
                                          inputCol='pca_input_features',
                                          outputCol='pca_features')
        """)
        return code.strip()


class SplitOperation(ModelMetaOperation):
    STRATEGY_TO_CLASS = {
        'split': 'CustomTrainValidationSplit',
        'cross_validation': 'CustomCrossValidation',
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.strategy = parameters.get('strategy', 'split')
        self.seed = parameters.get('seed', 'None') or 'None'
        self.ratio = parameters.get('ratio', 0.8) or 0.8

    def generate_code(self):
        if self.strategy == 'split':
            code = dedent(f"""
            train_ratio = {self.ratio} # Between 0.01 and 0.99
            executor = CustomTrainValidationSplit(
                pipeline, evaluator, grid, train_ratio, seed={self.seed})
            """)
        elif self.strategy == 'cross_validation':
            code = dedent(f"""
            """)

        return code.strip()


class GridOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.strategy = parameters.get('strategy', 'grid')
        self.random_grid = parameters.get(
            'random_grid') in ('1', 1, True, 'True')
        self.seed = parameters.get('seed')
        self.max_iterations = int(parameters.get('max_iterations', 0) or 0)
        self.max_search_time = int(parameters.get('max_search_time', 0) or 0)
        self.parallelism = int(parameters.get('parallelism', 0) or 0)

    def generate_code(self):
        code = []

        if self.strategy == 'grid':
            if self.random_grid:
                seed = f'.Random({self.seed})' if self.seed else ''
                code.append(f'random{seed}.shuffle(grid)')
            if self.max_iterations > 0:
                code.append(f'grid = grid[:{self.max_iterations}]')

        return '\n'.join(code)


class ClusteringOperation(EstimatorMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        EstimatorMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs, 'clustering')


class KMeansOperation(ClusteringOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClusteringOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = 'kmeans'
        self.types = parameters.get('type', ['kmeas'])

        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'initMode': _as_string_list(parameters.get('init_mode')),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info),
            'distanceMeasure': _as_string_list(parameters.get('distance')),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
        }
        self.name = 'KMeans'

    def get_variations(self):
        result = []
        if 'kmeans' in self.types:
            result.append(['KMeans', {}])
        if 'bisecting' in self.types:
            result.append(['BisectingKMeans', {'invalid': ['initMode']}])
        if len(result) == 0:
            result.append(['KMeans', {}])
        return result


class GaussianMixOperation(ClusteringOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClusteringOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = 'gaussian_mix'
        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info),
            'tol': _as_float_list(
                parameters.get('tolerance'), self.grid_info),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
        }
        self.name = 'GaussianMixture'


class ClassificationOperation(EstimatorMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        EstimatorMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs, 'classification')


class DecisionTreeClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'bootstrap': parameters.get('bootstrap'),
            'cacheNodeIds': _as_boolean_list(parameters.get('cache_node_ids')),
            'checkpointInterval':
            _as_int_list(parameters.get(
                'checkpoint_interval'), self.grid_info),
            'featureSubsetStrategy':
                _as_string_list(parameters.get('feature_subset_strategy')),
            'impurity': _as_string_list(parameters.get('impurity')),
            'leafCol': parameters.get('leaf_col'),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(
                parameters.get('max_depth'), self.grid_info),
            'maxMemoryInMB': parameters.get('max_memory_in_m_b'),
            'minInfoGain': _as_float_list(
                parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
            _as_int_list(parameters.get(
                'min_instances_per_node'), self.grid_info),
            'minWeightFractionPerNode':
                parameters.get('min_weight_fraction_per_node'),
            'numTrees': _as_int_list(
                parameters.get('num_trees'), self.grid_info),
            'seed': parameters.get('seed'),
            'subsamplingRate': parameters.get('subsampling_rate'),
            'weightCol': parameters.get('weight_col')
        }
        self.var = 'decision_tree'
        self.name = 'DecisionTreeClassifier'


class GBTClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'cacheNodeIds': parameters.get('cache_node_ids'),
            'checkpointInterval': parameters.get('checkpoint_interval'),
            'lossType': parameters.get('loss_type'),
            'maxBins': parameters.get('max_bins'),
            'maxDepth': _as_int_list(
                parameters.get('max_depth'), self.grid_info),
            'maxIter': _as_int_list(
                parameters.get('max_iter'), self.grid_info),
            'minInfoGain': _as_float_list(
                parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode': _as_int_list(
                parameters.get('min_instances_per_node'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'stepSize': parameters.get('step_size'),
            'subsamplingRate': parameters.get('subsampling_rate'),

        }
        self.var = 'gbt_classifier'
        self.name = 'GBTClassifier'


class NaiveBayesClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'modelType': parameters.get('model_type'),
            'smoothing': _as_float_list(parameters.get('smoothing')),
            'thresholds': parameters.get('thresholds'),
            'weightCol': parameters.get('weight_attribute'),
        }
        self.var = 'nb_classifier'
        self.name = 'NaiveBayes'


class PerceptronClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'layers': parameters.get('layers'),
            'blockSize': _as_int_list(
                parameters.get('block_size'), self.grid_info),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'solver': parameters.get('solver'),
        }
        self.var = 'mlp_classifier'
        self.name = 'MultilayerPerceptronClassifier'


class RandomForestClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'impurity': parameters.get('impurity'),
            'cacheNodeIds': parameters.get('cache_node_ids'),
            'checkpointInterval':
                parameters.get('checkpoint_interval'),
            'featureSubsetStrategy':
                _as_string_list(parameters.get('feature_subset_strategy')),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(
                parameters.get('max_depth'), self.grid_info),
            'minInfoGain': _as_float_list(
                parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
                parameters.get('min_instances_per_node'),
            'numTrees': _as_int_list(
                parameters.get('num_trees'), self.grid_info),
            'seed': parameters.get('seed'),
            'subsamplingRate': parameters.get('subsampling_rate'),
        }
        self.var = 'rand_forest_cls'
        self.name = 'RandomForestClassifier'


class LogisticRegressionOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'weightCol': _as_string_list(parameters.get('weight_col')),
            'family': _as_string_list(parameters.get('family')),
            'aggregationDepth': _as_int_list(
                parameters.get('aggregation_depth'), self.grid_info),
            'elasticNetParam': _as_float_list(
                parameters.get('elastic_net_param'), self.grid_info),
            'fitIntercept': _as_int_list(
                parameters.get('fit_intercept'), self.grid_info),
            'maxIter': _as_int_list(
                parameters.get('max_iter'), self.grid_info),
            'regParam': _as_float_list(
                parameters.get('reg_param'), self.grid_info),
            'tol': _as_float_list(parameters.get('tol'), self.grid_info),
            'threshold': _as_float_list(
                parameters.get('threshold'), self.grid_info),
            'thresholds': _as_string_list(parameters.get('thresholds')),
        }
        self.var = 'lr'
        self.name = 'LogisticRegression'


class SVMClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'standardization': _as_int_list(parameters.get('standardization'), self.grid_info),
            'threshold': _as_float_list(parameters.get('threshold'), self.grid_info),
            'tol': _as_float_list(parameters.get('tol'), self.grid_info),
            'weightCol': _as_string_list(parameters.get('weight_attr')),
        }
        self.var = 'svm_cls'
        self.name = 'LinearSVC'


class RegressionOperation(EstimatorMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        EstimatorMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs, 'regression')


class LinearRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        seed = 0
        max_iter = 5

        self.hyperparameters = {
            'aggregationDepth': parameters.get('aggregation_depth'),
            'elasticNetParam': _as_float_list(parameters.get('elastic_net'), self.grid_info),
            'epsilon': _as_float_list(parameters.get('epsilon'), self.grid_info),
            'fitIntercept': _as_boolean_list(parameters.get('fit_intercept')),
            'loss': _as_string_list(parameters.get('loss')),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'regParam': _as_float_list(parameters.get('reg_param'), self.grid_info),
            'solver': _as_string_list(parameters.get('solver')),
            'standardization': _as_boolean_list(parameters.get('standardization')),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'weightCol': parameters.get('weight'),
        }
        self.var = 'linear_reg'
        self.name = 'LinearRegression'


class IsotonicRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'isotonic': _as_boolean_list(parameters.get('isotonic')),
            'weightCol': parameters.get('weight'),
        }
        self.var = 'isotonic_reg'
        self.name = 'IsotonicRegression'


class GBTRegressorOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'cacheNodeIds': parameters.get('cache_node_ids'),
            'checkpointInterval': parameters.get('checkpoint_interval'),
            'featureSubsetStrategy': parameters.get('feature_subset_strategy'),
            'impurity': parameters.get('impurity'),
            'leafCol': parameters.get('leaf_col'),
            'lossType': parameters.get('loss_type'),
            'maxBins': parameters.get('max_bins'),
            'maxDepth': _as_int_list(parameters.get('max_depth'), self.grid_info),
            'maxIter': parameters.get('max_iter'),
            'maxMemoryInMB': parameters.get('max_memory_in_m_b'),
            'minInfoGain': _as_float_list(parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode': _as_int_list(parameters.get('min_instance'), self.grid_info),
            'minWeightFractionPerNode':
                parameters.get('min_weight_fraction_per_node'),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'stepSize': parameters.get('step_size'),
            'subsamplingRate': parameters.get('subsampling_rate'),
            'validationIndicatorCol':
            parameters.get('validation_indicator_col'),
            'validationTol': parameters.get('validation_tol'),
            'weightCol': parameters.get('weight_col')

        }
        self.var = 'gbt_reg'
        self.name = 'GBTRegressor'


class DecisionTreeRegressorOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(parameters.get('max_depth'), self.grid_info),
            'minInfoGain': _as_float_list(parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
                parameters.get('min_instances_per_node'),
        }
        self.var = 'dt_reg'
        self.name = 'DecisionTreeRegressor'


class RandomForestRegressorOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'bootstrap': parameters.get('bootstrap'),
            'cacheNodeIds': parameters.get('cache_node_ids'),
            'checkpointInterval':
                parameters.get('checkpoint_interval'),
            'featureSubsetStrategy':
                _as_string_list(parameters.get('feature_subset_strategy')),
            'impurity': parameters.get('impurity'),
            'leafCol': parameters.get('leaf_col'),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(parameters.get('max_depth'), self.grid_info),
            'maxMemoryInMB': parameters.get('max_memory_in_m_b'),
            'minInfoGain': _as_float_list(parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
                parameters.get('min_instances_per_node'),
            'minWeightFractionPerNode':
                parameters.get('min_weight_fraction_per_node'),
            'numTrees': _as_int_list(parameters.get('num_trees'), self.grid_info),
            'seed': parameters.get('seed'),
            'subsamplingRate': parameters.get('subsampling_rate'),
            'weightCol': parameters.get('weight_col')
        }
        self.var = 'rand_forest_reg'
        self.name = 'RandomForestRegressor'


class GeneralizedLinearRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.family_link = parameters.get('family_link') or []
        if 'solver' in parameters:
            parameters['solver'] = [s for s in parameters['solver']
                                    if s != 'auto']

        self.hyperparameters = {
            # 'aggregationDepth': parameters.get('aggregation_depth'),
            # 'fitIntercept': parameters.get('fit_intercept'),
            # 'linkPower': parameters.get('link_power'),
            # 'maxIter': parameters.get('max_iter'),
            # 'offsetCol': parameters.get('offset'),
            'regParam': _as_float_list(parameters.get('elastic_net'),
                                       self.grid_info),
            'solver': _as_string_list(parameters.get('solver')),
            # 'standardization': parameters.get('standardization'),
            # 'tol': parameters.get('tol'),
            # 'variancePower': parameters.get('variance_power'),
            # 'weightCol': parameters.get('weight'),
        }
        self.var = 'gen_linear_regression'
        self.name = 'GeneralizedLinearRegression'

    def get_constrained_params(self):
        result = []
        for family, link in [x.split(':') for x in self.family_link]:
            result.append(
                f'{{{self.var}.family: {repr(family)}, '
                f'{self.var}.link: {repr(link)}}}')
        return result


class VisualizationOperation(MetaPlatformOperation):

    DEFAULT_PALETTE = ['#636EFA', '#EF553B', 
        '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
        '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.type = self.get_required_parameter(parameters, 'type')
        self.display_legend = self.get_required_parameter(parameters, 'display_legend')
        self.palette = parameters.get('palette')
        self.x = self.get_required_parameter(parameters, 'x')
        self.y = self.get_required_parameter(parameters, 'y')
        self.x_axis = self.get_required_parameter(parameters, 'x_axis')
        self.y_axis = self.get_required_parameter(parameters, 'y_axis')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            k: {'value': getattr(self, k)} for k in 
                ['type', 'display_legend', 'palette', 'x', 'y', 'x_axis', 'y_axis']
        })
        task_obj['operation'] = {"id": 145}

        return json.dumps(task_obj)

class BatchMetaOperation(Operation):
    """ Base class for batch execution operations """
    def __init__(self, parameters,  named_inputs, named_outputs):
       super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self) -> str:
        return ''

class ConvertDataSourceFormat(BatchMetaOperation):
    LIMONERO_TO_SPARK_DATA_TYPES = {
        "BINARY": 'types.BinaryType',
        "CHARACTER": 'types.StringType',
        "DATETIME": 'types.TimestampType',
        "DATE": 'types.DateType',
        "DOUBLE": 'types.DoubleType',
        "DECIMAL": 'types.DecimalType',
        "FLOAT": 'types.FloatType',
        "LONG": 'types.LongType',
        "INTEGER": 'types.IntegerType',
        "TEXT": 'types.StringType',
    }

    SUPPORTED_DRIVERS = {
        'mysql': 'com.mysql.jdbc.Driver'
    }
    DATA_TYPES_WITH_PRECISION = {'DECIMAL'}

    SEPARATORS = {
        '{tab}': '\\t',
        '{new_line \\n}': '\n',
        '{new_line \\r\\n}': '\r\n'
    }
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    QUOTE_PARAM = 'quote'
    INFER_SCHEMA_PARAM = 'infer_schema'
    NULL_VALUES_PARAM = 'null_values'

    INFER_FROM_LIMONERO = 'FROM_LIMONERO'
    INFER_FROM_DATA = 'FROM_VALUES'


    template = """
        {%- if infer_schema == 'FROM_LIMONERO' %}
        # Schema definition
        schema = types.StructType()
        {%- for attr in attributes %}
        schema.add('{{attr.name}}', {{attr.data_type}}, {{attr.nullable}})
        {%- endfor %} 
        {%- elif infer_schema == 'FROM_DATA' %}
        schema = None
        {%- endif %}

        # Open the source data
        url = '{{url}}' #protect: url
        df = (spark_session
            .read{{null_option}}
            .option('treatEmptyValuesAsNulls', 'true')
            .option('wholeFile', True)
            .option('multiLine', {{multiline}})
            .option('escape', '"')
            .option('timestampFormat', '{{date_fmt}}')
            .csv(
                url, schema=schema,
                {%- if quote %}
                quote='{{quote}}',
                {%- endif %}
                ignoreTrailingWhiteSpace=True, # Handle \\r
                encoding='{{encoding}}',
                header={{header}}, 
                sep='{{sep}}',
                inferSchema={{infer_schema == 'FROM_DATA'}},
                mode='IGNORE'
            )
        )
        target_url = f'{url}.parquet' #protect: url
        df.repartition(1).write.mode('overwrite').parquet(target_url)

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)
        
        self.data_source_id = self.get_required_parameter(
            parameters, 'data_source')
        self.target_format = self.get_required_parameter(
            parameters, 'target_format')
        self.parameters = parameters
        self.has_code = True

        self.transpiler_utils.add_import(
            'from pyspark.sql import functions, types, Row, DataFrame')

    def generate_code(self) -> str:

        parameters = self.parameters
        # Load metadata
        limonero_config = \
            self.parameters['configuration']['juicer']['services'][
                'limonero']
        url = limonero_config['url']
        token = str(limonero_config['auth_token'])
        metadata = limonero_service.get_data_source_info(
            url, token, self.data_source_id)

        if metadata.get('format') not in ('CSV', ):
            raise ValueError(
                gettext('Unsupported format: {}').format(
                    metadata.get('format')))

        header = (metadata.get('is_first_line_header', False) 
            not in ('0', 0, 'false', False))
        null_values = [v.strip() for v in parameters.get(
            self.NULL_VALUES_PARAM, '').split(",")]

        record_delimiter = metadata.get('record_delimiter')
        sep = parameters.get(
            self.SEPARATOR_PARAM,
            metadata.get('attribute_delimiter', ',')) or ','
        quote = parameters.get(self.QUOTE_PARAM,
                                    metadata.get('text_delimiter'))
        if quote == '\'':
            quote = '\\\''
        if sep in self.SEPARATORS:
            sep = self.SEPARATORS[sep]
        infer_schema = parameters.get(self.INFER_SCHEMA_PARAM,
                                      self.INFER_FROM_LIMONERO)

        encoding = metadata.get('encoding', 'UTF-8') or 'UTF-8'

        if metadata.get('treat_as_missing'):
            null_values.extend([x.strip() for x in 
                metadata.get('treat_as_missing').split(',')])
        null_option = ''.join(
            [f".option('nullValue', '{n}')" for n in
             set(null_values)]) if null_values else ""
        url = metadata.get('url')

        if infer_schema == self.INFER_FROM_LIMONERO:
            for attr in metadata.get('attributes'):
                data_type = self.LIMONERO_TO_SPARK_DATA_TYPES[attr['type']]
                if attr['type'] in self.DATA_TYPES_WITH_PRECISION:
                    # extra precision
                    # precision = (attr.get('precision', 0) or 0) + 3
                    # scale = (attr.get('scale', 0) or 0) or 0
                    # data_type = f'{data_type}({precision}, {scale})'

                    # Polars does not support Decimal
                    # See https://github.com/pola-rs/polars/issues/4104
                    data_type = 'types.FloatType()'
                else:
                    data_type = f'{data_type}()'
                attr['data_type'] = data_type



        ctx = {
            'attributes': metadata.get('attributes'),
            'date_fmt': "yyyy/MM/dd HH:mm:ss",
            'encoding': encoding,
            'header': header,
            'infer_schema': infer_schema,
            'multiline': encoding in ('UTF-8', 'UTF8', ''),
            'null_option': null_option,
            'op': self,
            'quote': quote,
            'record_delimiter': record_delimiter,
            'sep': sep,
            'url': url,
        }
        return dedent(self.render_template(ctx))
