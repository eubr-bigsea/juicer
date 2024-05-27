import dataclasses
import json
import re
from collections import namedtuple
from functools import reduce
from gettext import gettext
from itertools import zip_longest as zip_longest
from textwrap import dedent, indent
from typing import List

from juicer.operation import Operation
from juicer.service import limonero_service
from juicer.spark.data_operation import DataReaderOperation
from juicer.spark.data_operation import SaveOperation as SparkSaveOperation
from juicer.spark.etl_operation import AggregationOperation, SampleOrPartitionOperation
from juicer.spark.etl_operation import FilterOperation as SparkFilterOperation
from juicer.util.template_util import strip_accents

FeatureInfo = namedtuple('FeatureInfo', ['value', 'props', 'type'])

convert_re = re.compile(r'np.linspace\((\d+), (\d+), \d+, dtype=int\)')
@dataclasses.dataclass
class HyperparameterInfo:
    value: any
    values_count: int
    param_type: str
    random_generator: any = None

    def __str__(self):
        if self.param_type == 'list':
            return repr(self.value)
        elif self.param_type == 'range':
            return repr(self.value)
        else:
            return '<INVALID hyperparameter type>'



non_word_chars_pattern = re.compile(r'[^a-zA-Z0-9_]+', re.U)

def pythonize(s: str) -> str:
    """
    Pythonize a string by removing accents and replacing non-word characters
    with underscores.

    Args:
        s (str): The input string.
    Returns:
        str: The pythonized string.
    """
    return non_word_chars_pattern.sub('_', strip_accents(s))


def _as_list(input_values, transform=None, size=None, validate=None):
    """
    About log linear:
    https://towardsdatascience.com/why-is-the-log-uniform-distribution-useful-for-hyperparameter-tuning-63c8d331698
    """
    if (input_values is None or not isinstance(input_values, dict) or
            (not input_values.get('enabled', True))):
        return None
    param_type = input_values.get('type')

    def apply(values):
        if param_type == 'list':
            values = values.get('list')
            return (values, f'np.random.choice({values})')
        elif param_type == 'range':
            qty = size or values.get('quantity') or values.get('size', 3)
            _min = values.get('min', 0)
            _max = values.get('max', 3)
            if values.get('distribution') == 'log_uniform':
                # Values must be greater than 0
                _min2 = max(_min, 1e-10)
                return (
                    f'np.logspace('
                    f'np.log10({_min2}), np.log10({_max}), {qty}).tolist()',
                    f'np.random.uniform({_min}, {_max} + 1e-10)'
                )
            else:
                return (
                    f'np.linspace('
                    f'{_min}, {_max}, {qty}, dtype=int).tolist()',
                    f'np.random.randint({_min}, {_max} + 1)')

    result, random_generator = apply(input_values)
    if param_type == 'list' and len(result) == 0:
        return None
    if param_type == 'list':
        if transform:
            if validate:
                _validate(validate, result)
            transformed = [transform(x) for x in result if x is not None]
            return HyperparameterInfo(
                value=transformed, param_type='list', values_count=len(transformed),
                random_generator=random_generator)
        else:
            return HyperparameterInfo(
                value=result, param_type='list', values_count=len(result),
                random_generator=random_generator)

    qty = size or input_values.get('quantity') or input_values.get('size', 3)
    return HyperparameterInfo(
        value=result, param_type='range', values_count=qty,
        random_generator=random_generator)

    # if input_values:
    #     if isinstance(input_values, list):
    #         return FeatureInfo(input_values, None, 'simple_list')
    #     elif isinstance(input_values, str):
    #         return []
    #     elif (input_values.get('type') == 'list' or (
    #             input_values.get('type') != 'range' and input_values.get('list'))):
    #         v = [transform(x) for x in input_values.get(
    #             'list', []) if x is not None]
    #         return FeatureInfo(v, input_values, 'list')
    #     elif input_values.get('type') == 'range':
    #         qty = size or input_values.get('quantity') or 3
    #         _min = input_values.get('min', 0)
    #         _max = input_values.get('max', 3)
    #         if input_values.get('distribution') == 'log_uniform':
    #             return FeatureInfo(
    #                 f'random.sample(np.log10(np.logspace('
    #                 f'{_min}, {_max}, {qty})).tolist(), {qty})',
    #                 input_values, 'function')
    #         else:
    #             return FeatureInfo(
    #                 f'random.sample(np.linspace('
    #                 f'{_min}, {_max}, {qty}, dtype=int).tolist(), {qty})',
    #                 input_values, 'function')
    #     elif input_values.get('type') is None:
    #         return []
    #     else:
    #         return FeatureInfo(
    #             [transform(x) for x in input_values if x is not None], None)
    # else:
    #     return []


def _as_boolean_list(values):
    if values is None:
        return None
    values['list'] = [v for v in values['list'] if v in [False, True]]
    return _as_list(values)

'''
def _as_int_list(values, grid_info=None, validate=None):
    return _as_list(values, int, validate=validate)
    size = None
    if 'value' in grid_info:
        value = grid_info.get('value')
        if value.get('strategy') == 'random':
            size = value.get('max_iterations')
    return _as_list(values, int, size)
'''
def _as_int_list(values, grid_info=None, validate=None):
    size = None
    if grid_info and 'value' in grid_info:
        value = grid_info['value']
        if value.get('strategy') == 'random':
            size = value.get('max_iterations')
    return _as_list(values, int, size, validate=validate)


def _as_float_list(values, grid_info, validate=None):
    result = _as_list(values, transform=float, validate=validate)
    return result

    # size = None
    # if 'value' in grid_info:
    #     value = grid_info.get('value')
    #     if value.get('strategy') == 'random':
    #         size = value.get('max_iterations')

    # return _as_list(values, float, size)


def _as_string_list(values, validate=None):
    result = _as_list(values, str, validate=validate)
    return result


def _validate(validate, result):
    if result and validate:
        invalid = validate(result)
        if invalid:
            raise ValueError(
                gettext('Invalid value(s) for parameter: {}').format(
                    repr(invalid)))


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
                    "display_text": {"value": "0"}
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
    TARGET_OP = 16
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
        params['connection_factory_function_name'] = 'get_hwc_connection'
        dro = DataReaderOperation(params, {}, {'output data': 'df'})
        return dro.generate_code()

    def visualization_builder_code(self):
        return self.model_builder_code()

    def sql_code(self):
        return self.model_builder_code()

class ExecuteSQLOperation(MetaPlatformOperation):
    TARGET_OP = 93
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.query = self.get_required_parameter(
            parameters, 'query')
        self.input_port_name = 'input data 1'
        valid_true = (1, '1', 'true', True)
        self.save = parameters.get('save') in valid_true
        self.use_hwc = parameters.get('useHWC', False) in valid_true
        if self.save:
            self.transpiler_utils.add_import(
                'from juicer.service.limonero_service import register_datasource')

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            "query": {"value": self.query},
        })
        task_obj['operation'] = {"id": self.TARGET_OP}
        return json.dumps(task_obj)

    def _not_first(self):
        """Creates a function returning False only the first time."""
        _first_time_call = True

        def fn(_) -> bool:
            nonlocal _first_time_call

            res = not _first_time_call
            _first_time_call = False
            return res

        return fn
    def sql_code(self):
        sql = repr(self.query.strip())[1:-1].replace('\\n', '\n').replace(
            '"""', '')
        code = []

        if self.use_hwc:
            cmd = 'get_hwc_connection(spark_session).execute(sql)'
        else:
            cmd = 'spark_session.sql(sql)'
        code.append(dedent(
            f"""
            sql = \"\"\"
                {indent(dedent(sql), ' '*15, self._not_first())}
            \"\"\"
            result = {cmd}
            """).strip())

        if self.save:
            params = {}
            params.update(self.parameters)
            params['name'] = self.parameters.get('new_name')
            params['format'] = 'PARQUET' # FIXME
            params['path'] = params.get('path', '')

            dro = SparkSaveOperation(params, {'input data': 'df'}, {})
            code.append(dro.generate_code())
        return '\n'.join(code)

@dataclasses.dataclass
class TransformParam:
    function: str
    args: List[any] = dataclasses.field(default_factory=list)
    transform: List[callable] = dataclasses.field(default_factory=list)

class TransformOperation(MetaPlatformOperation):
    TARGET_OP = 7
    number_re = r'([\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+)'
    array_index_re = re.compile(r'(?:\D*?)(-?\d+)(?:\D?)')
    SLUG_TO_EXPR = {
        # 'extract-numbers': Transform()'regexp_extract', [number_re, 1],
        #                    [str, None]),
        'extract-numbers': TransformParam('extract_numbers', [], []),
        'extract-with-regex': TransformParam('regexp_extract', ['{regex}'],
                                             [str]),
        'replace-with-regex': TransformParam(
            'regexp_replace', ['{regex}', '{replace}'],
            [str, lambda v: '' if v is None else v]),
        'to-upper': TransformParam('upper', None, None),
        'to-lower': TransformParam('lower', None, None),
        'capitalize': TransformParam('initcap', None, None),
        'remove-accents': TransformParam('strip_accents', None, None),
        'parse-to-date': TransformParam('to_date', ['{format}'], [str]),
        'split': TransformParam('split', None, None),
        'trim': TransformParam('trim', None, None),
        'normalize': TransformParam('FIXME', None, None),
        'regexp_extract': TransformParam('regexp_extract', ['{delimiter}'],
                                         [str]),
        'round-number': TransformParam('round', ['{decimals}'], [int]),
        'split-into-words': TransformParam('split', ['{delimiter}'], [str]),
        'truncate-text': TransformParam('substring', ['0', '{characters}'],
                                        [int, int]),
        'ts-to-date': TransformParam('from_unixtime', None, None),
        'date-to-ts': TransformParam('unix_timestamp', None, None),
        'date-part': TransformParam('None', None, None),
        'format-date': TransformParam('date_format', ['{format}'], [str]),
        'truncate-date-to': TransformParam('date_trunc', ['{format}'], [str]),
        #'invert-boolean': TransformParam('None', None, None),
        #'extract-from-array': TransformParam('None', None, None),
        'concat-array': TransformParam('array_join', ['{delimiter}'], [str]),
        'sort-array': TransformParam('array_sort', None, None),
        'change-array-type': TransformParam('array_cast', ['{new_type}'],
                                            [str]),
        'flag-empty': TransformParam('isnull', None, None),
        #'flag-with-formula': TransformParam('None', None, None),
    }
    SUPPORTED_FUNCTIONS = list(SLUG_TO_EXPR.keys()) + [
        'invert-boolean', 'date-add', 'date-part', 'extract-from-array',
        'flag-with-formula']
    ALIASES = {
        'flag-empty': '_na'
    }
    DATE_COMPONENT_2_FN = {
        'second': 'seconds',
        'minute': 'minutes',
        'hour': 'hours',
        'day': 'days',
        'week': 'weeks',
        'month': 'months',
        'year': 'years',
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        op = parameters.get('task').get('operation')
        self.slug = op.get('slug')

        if self.slug not in self.SUPPORTED_FUNCTIONS:
            raise ValueError(gettext('Invalid function {} in transformation')
                             .format(self.slug))
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        if not self.attributes:
            raise ValueError(gettext('Missing required parameter: {}').format(
                'attributes'))

    def generate_code(self):
        task_obj = self._get_task_obj()

        if self.slug in self.SLUG_TO_EXPR:
            info = self.SLUG_TO_EXPR[self.slug]
            function_name = info.function
        else:
            function_name = None
        alias = self.ALIASES.get(self.slug, '')
        expressions = []
        if function_name is not None:
            self.form_parameters = {}
            param_names = []
            for i, (arg, transform) in enumerate(zip(info.args or [],
                                                     info.transform or [])):
                if transform is not None:
                    if arg[0] == '{' and arg[-1] == '}':
                        param_name = arg[1:-1]
                        self.form_parameters[param_name] = transform(
                            self.get_required_parameter(self.parameters,
                                                        param_name))
                        param_names.append(param_name)
                    else:
                        self.form_parameters[f'param_{i}'] = transform(arg)
                        param_names.append(f'param_{i}')
            # Convert the parameters
            args = info.args or []
            function_args = [
                (arg.format(**self.form_parameters)
                 if isinstance(arg, str) else str(arg)) for arg in args]

            final_args_str = ''
            final_args = []

            if function_args:
                final_args_str = ', ' + ', '.join(function_args)
                transform = info.transform
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
                            'arguments': [{'type': 'Identifier', 'name': attr}
                                          ] + final_args,
                            'callee': {'type': 'Identifier',
                                       'name': function_name},
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


            component = self.parameters.get('component', 'day')
            f = self.DATE_COMPONENT_2_FN.get(component)
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
            f = self.DATE_COMPONENT_2_FN.get(component)
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
                            'callee': {'type': 'Identifier',
                                       'name': 'element_at'},
                        }
                    })

        elif self.slug == 'flag-with-formula':
            expressions.extend(self.parameters.get('formula'))

        task_obj['forms']['expression'] = {'value': expressions}
        task_obj['operation'] = {'id': self.TARGET_OP}
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
    TARGET_OP = 28

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.type = parameters.get('type')
        self.value = int(parameters.get('value', 0) or 0)
        self.seed = parameters.get('seed')
        self.fraction = parameters.get('fraction')
        if self.fraction is not None:
            self.fraction = float(self.fraction)
        self.output_port_name = 'sampled data'
        self.has_code = (
            (self.value is not None and self.value > 0) or
                (self.fraction is not None and self.fraction > 0))

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            'type': {'value': self.type},
            'value': {'value': self.value},
            'fraction': {'value':
                         100 * self.fraction if self.fraction is not None
                         else None},
            'seed': {'value': self.seed},
        })
        task_obj['operation'] = {"id": self.TARGET_OP}
        return json.dumps(task_obj)

    def model_builder_code(self):
        spo = SampleOrPartitionOperation(self.parameters, {'input data': 'df'},
                                         {'sampled data': 'df'})
        return spo.generate_code()

    def visualization_builder_code(self):
        params = {}
        params.update(self.parameters)
        if 'fraction' in params:
            params['fraction'] *= 100 # Implementation requires values in percent
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
                        "type": "CallExpression",
                        "arguments": [
                            {
                                "type": "BinaryExpression",
                                "operator": "|",
                                "left": {
                                    "type": "BinaryExpression",
                                    "operator": "<",
                                    "left": {
                                        "type": "Identifier",
                                        "name": attr
                                    },
                                    "right": {
                                        "type": "Literal",
                                        "value": self.start,
                                        "raw": self.start
                                    }
                                },
                                "right": {
                                    "type": "BinaryExpression",
                                    "operator": ">",
                                    "left": {
                                        "type": "Identifier",
                                        "name": attr
                                    },
                                    "right": {
                                        "type": "Literal",
                                        "value": self.end,
                                        "raw": self.end
                                    }
                                }
                            },
                            {
                                "type": "Literal",
                                "value": None,
                                "raw": None
                            },
                            {
                                "type": "Identifier",
                                "name": attr
                            }
                        ],
                        "callee": {"type": "Identifier", "name": "when"}}
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
                        "type": "CallExpression",
                        "arguments": [
                            {
                                "type": "BinaryExpression",
                                "operator": "<",
                                "left": {
                                    "type": "Identifier",
                                    "name": attr
                                },
                                "right": {
                                    "type": "Literal",
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
                                "type": "BinaryExpression",
                                "operator": ">",
                                "left": {
                                    "type": "Identifier",
                                    "name": attr
                                },
                                "right": {
                                    "type": "Literal",
                                    "value": self.end,
                                    "raw": self.end
                                }
                            },
                            {
                                "type": "Literal",
                                "value": self.end,
                                "raw": self.end
                            },
                            {
                                "type": "Identifier",
                                "name": attr
                            }
                        ],
                        "callee": {"type": "Identifier", "name": "when"}}
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
        self.formula = parameters.get('formula')
        self.has_code = bool(self.formula)

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
        """ FIXME Code is not correct"""
        task_obj = self._get_task_obj()
        function_name = ''
        attr = ''
        final_args_str = ''
        final_args = []
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
            'formula': formula
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
                                'mode', 'header', 'storage', 'description',
                                'data_source_id']

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
    TARGET_OP = 18
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
            task_obj['forms'][param] = {
                'value': self.parameters.get(param),
            }

        read_task = {
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

        return json.dumps(read_task) + ',' + json.dumps(task_obj)

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

    def model_builder_code(self):
        return self.generate_code()


class EstimatorMetaOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs, task_type):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = "CHANGE_VAR"
        self.name = "CHANGE_NAME"
        self.hyperparameters: dict[str, HyperparameterInfo] = {}
        self.task_type = task_type
        #self.grid_info = parameters.get('workflow').get(
            #'forms', {}).get('$grid', {})
        self.grid_info = parameters.get('workflow', {}).get('forms', {}).get('$grid', {})


    def get_constrained_params(self):
        return None

    def generate_code(self):
        code = []
        name = self.name
        var = self.var
        variations = self.get_variations()

        if len(variations) > 0:
            for i, (klass, params) in enumerate(variations):
                operation_id = self.task.get('operation').get('id')
                code.append(dedent(f"""
                    {var}_{i} = {self.task_type}.{klass}(**common_params)

                    # Lemonade internal use
                    {var}_{i}.task_id = '{self.task.get('id')}'
                    {var}_{i}.task_name = '{self.task.get('name')} ({klass})'
                    {var}_{i}.operation_id = {operation_id}
                    """))
        else:
            code.append(dedent(f"""
                {var} = {self.task_type}.{name}(**common_params)

                # Lemonade internal use
                {var}.task_id = '{self.task.get('id')}'
                {var}.task_name = '{self.task.get('name')}'
                {var}.operation_id = {self.task.get('operation').get('id')}
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
        max_iterations = self.grid_info.get('value', {}).get(
            'max_iterations', 20)
        self.template = """
            grid_{{var}} = (tuning.ParamGridBuilder()
                .baseOn({pipeline.stages: common_stages + [{{var}}] })
                {%- for p, v in hyperparameters %}
                .addGrid({{var}}.{{p}}, {{v.value}})
                {%- endfor %}
                .build()
            )
            {%- if strategy == 'random' %}
            grid_{{var}} = []
            found = 0
            attemps = 0
            for _ in range(2*{{max_iterations}}): #FIXME!! Double generation in order to fill when coliding
                # Use seed!
                params = []
                for p, v in rand_{{var}}.items():
                    params.append((p, random.choice(v)))
                grid_{{var}}.append(dict(params))

            {% endif -%}
        """
        return (dedent(self.render_template({
            'strategy': grid_strategy,
            'var': var, 'max_iterations': max_iterations,
            'hyperparameters': [(p, v) for p, v in self.hyperparameters.items()
                                if v]
        }))).strip()
        print('*' * 20)
        if grid_strategy in ('grid', 'random'):
            for name, value in self.hyperparameters.items():
                if value is not None:
                    code.append(f'.addGrid({var}.{name}, {value})')
                continue
                if value is not None and value and hasattr(value, 'value') and len(value.value) > 0:
                    if name not in invalid:
                        if grid_strategy == 'grid' or value.type == 'simple_list':
                            v = self.parse(value)
                            if v and v != '[]':
                                code.append(
                                    f'.addGrid({var}.{name}, {v})')
                elif isinstance(value, dict):
                    if grid_strategy == 'grid' or value.type == 'simple_list':
                        v = self.parse({'value': value})
                        code.append(
                            f'.addGrid({var}.{name}, {v})')

            if grid_strategy == 'grid':
                code.append('.build()')
            else:
                code.append('._param_grid # Uses internal variable')
        else:
            raise ValueError(gettext('Invalid grid search strategy: {}').format(
                grid_strategy))
        return dedent('\n'.join(code))

    def generate_random_hyperparameters_code(self, limit=10, seed=None):

        n = min(limit, self.get_hyperparameter_count())

        code = []
        if seed:
            code.append(f'np.random.seed({seed})')
        self.template = """
            {%- if seed %}
            np.random.seed({seed})
            {%- endif %}
            n = {{n}}
            grid_{{var}} = []
            # generated_param_values = set()
            for i in range(n):
                param_dict = {
                {%- for name, value in params.items() %}
                    {{var}}.{{name}}: {{value.random_generator}},
                {%- endfor %}
                }
                #generated_param_values.add(tuple(param_dict.values())
                grid_{{var}}.append(param_dict)

        """
        ctx = {
            'var': self.var, 'seed': seed, 'n': n,
            'params': dict((k, v) for k, v in self.hyperparameters.items() if v)
        }
        return dedent(self.render_template(ctx))

        code.append('grid = [')
        for i in range(n):
            code.append('    {')
            for name, value in self.hyperparameters.items():
                code.append(f'        {self.var}.{name}: ""')
            code.append('    },')
        code.append(']')
        return dedent('\n'.join(code))

        grid_strategy = self.grid_info.get('value', {}).get('strategy', 'grid')
        random_hyperparams = []
        for name, value in self.hyperparameters.items():
            if value is not None and value and hasattr(value, 'value') and len(value.value) > 0:
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

    def get_hyperparameter_count(self):
        return reduce(lambda x, param: x * param.values_count,
                      [v for v in self.hyperparameters.values() if v], 1)

    def parse(self, v):
        value = v
        # if isinstance(v, FeatureInfo) and v.type == 'list':
        #     return v.props.get('list')
        # if isinstance(v, FeatureInfo) and v.type == 'function':
        #     return value.value
        # else:
        #     value = v.get('value', {})
        #     if value.get('type') == 'list':
        #         return repr([x for x in value.get('list') if x is not None])
        #     else:
        #         raise ValueError(
        #             gettext('Invalid parameter type: {} for {}'.format(
        #                 value.get('type'), value)))
        #
        if isinstance(value, list):
            return repr([x.value for x in value if x is not None and not isinstance(x, dict)])
        elif isinstance(value, FeatureInfo) and value.type == 'function':
            return value.value
        elif hasattr(value, 'type') and value.type == 'list':
            new_value = [v for v in value.value if not isinstance(v, dict) and
                         (not isinstance(v, str) or not '{' in v)]
            return repr(new_value)
        elif isinstance(value, dict) and 'value' in value:
            return repr(value.get('value'))
        return repr(value.value)

    def get_variations(self):
        return []

    def ge(self, compare_to):
        return lambda x: [v for v in x if v < compare_to]

    def gt(self, compare_to):
        compare_to_int = int(compare_to)
        return lambda x: [v for v in x if v <= compare_to]

    '''
    def gt(self, compare_to):
        compare_to_str = str(compare_to)
        return lambda x: [v for v in x if v <= compare_to_str]
    '''
    '''
    def gt(self, compare_to):
        return lambda x: [int(v) if isinstance(v, str) else v for v in x if isinstance(v, int) or (isinstance(v, str) and isinstance(compare_to, str) )]
    '''

    def between(self, start, end, include_start=True, include_end=True):
        # Define the comparison operators based on inclusion options
        start_comp = ((lambda a, b: a <= b)
                      if include_start else (lambda a, b: a < b))
        end_comp = ((lambda a, b: a <= b) if
                    include_end else (lambda a, b: a < b))

        # Filter the input list based on the defined conditions
        return (lambda x: [v for v in x if start_comp(start, v)
                           and end_comp(v, end)])

    def in_list(self, *search_list):
        return lambda x: [v for v in x if v not in search_list]


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
        # meta_form = self.parameters.get('workflow').get(
        #     'forms', {}).get('$meta', {}).get('value')

        # if meta_form.get('taskType') != 'clustering':
        #     label = 'labelCol=label'
        # else:
        #     label = ''  # clustering doesn't support it
        if self.task_type in ('binary-classification', 'regression',
                              'multiclass-classification'):
            label = 'labelCol=label'
        else:
            label = ''  # clustering doesn't support it

        code = dedent(
            f"""
            evaluator = evaluation.{evaluator}(
                metricName='{self.metric}', {label})
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
    __slots__ = ('all_attributes', 'label', 'features', 'features',
                 'categorical_features', 'textual_features', 'features_names'
                 'features_and_label', 'task_type', 'supervisioned'
                 )

    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.features_and_label = parameters.get('features', [])
        meta_form = self.parameters.get('workflow').get(
            'forms', {}).get('$meta', {}).get('value')
        self.task_type = meta_form.get('taskType')
        self.process_features_and_label()

    def process_features_and_label(self):
        self.all_attributes = [f for f in self.features_and_label
                               if f.get('enabled')]
        self.features = []
        self.numerical_features = []
        self.categorical_features = []
        self.textual_features = []
        self.features_names = []
        self.label = None
        for f in self.features_and_label:
            if f.get('usage') == 'unused' or not f.get('usage'):
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

        if self.task_type != 'clustering' and self.label is None:
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
        label = self.label.get('name') if self.label else ''

        for f in self.features_and_label:
            usage = f.get('usage')
            if usage == 'unused' or not usage:
                continue
            name = f.get('name')
            transform = f.get('transform')
            data_type = f.get('feature_type')
            missing = f.get('missing_data')
            scaler = f.get('scaler')
            is_numerical = data_type == 'numerical'

            if f.get('feature_type') not in ('numerical', 'categorical',
                                             'textual', 'vector'):
                raise ValueError(gettext(
                    "Invalid feature type '{}' for attribute '{}'."
                ).format(transform, f.get('name')))

            if f.get('usage') not in ('label', 'feature', 'unused'):
                raise ValueError(gettext(
                    "Invalid feature usage '{}' for attribute '{}'."
                ).format(f.get('usage'), f.get('name')))

            final_name = name
            # Handle missing
            # if f.get('name') == 'sepallength':
            #    import pdb; pdb.set_trace()
            if missing == 'constant' and transform != 'not_null':
                cte = f.get('constant')
                if cte is None:
                    raise ValueError(gettext(
                        "Missing constant value for attribute '{}'."
                    ).format(f.get('name')))
                if data_type == 'categorical':
                    stmt = (f"SELECT *, COALESCE({f['name']}, '{cte}') AS "
                            f"{f['var']}_na FROM __THIS__")
                elif is_numerical:
                    stmt = (f"SELECT *, COALESCE({f['name']}, {cte}) AS "
                            f"{f['var']}_na FROM __THIS__")

                code.append(dedent(f"""
                    {f['var']}_na = feature.SQLTransformer(
                        statement="{stmt}")
                    features_stages.append({f['var']}_na) """))
                f['na_name'] = f['name']
                final_name = name + '_na'
            elif missing == 'remove' and transform != 'not_null':
                stmt = f"SELECT * FROM __THIS__ WHERE ({f['name']}) IS NOT NULL"
                code.append(dedent(f"""
                    {f['var']}_del_nulls = feature.SQLTransformer(
                        statement="{stmt}")
                    features_stages.append({f['var']}_del_nulls) """))
                f['na_name'] = f['name']
                final_name = name
            elif missing in ('median', 'media') and is_numerical:
                # import pdb; pdb.set_trace()
                code.append(dedent(f"""
                    {name}_imp = feature.Imputer(
                        strategy='{missing}',
                        inputCols=['{name}'],
                        outputCols=['{name}_na'])
                    features_stages.append({name}_na)
                """).strip())
                final_name = name + '_na'

            if is_numerical:
                if transform in ('keep', '', None):
                    #final_name = name
                    ...
                elif transform == 'binarize':
                    final_name = final_name + '_bin'
                    threshold = self.parameters.get('threshold', f['threshold'])
                    code.append(dedent(f"""
                        {f['var']}_bin = feature.Binarizer(
                            threshold={threshold}, inputCol='{f['na_name']}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_bin) """))
                elif transform in ('quantiles', 'quantis'):
                    final_name = final_name + '_qtles'
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
                        final_name = final_name + '_bkt'
                        code.append(dedent(f"""
                            {f['var']}_qtles = feature.Bucketizer(
                                splits=[-float('inf'), {splits}, float('inf')],
                                inputCol='{f['na_name']}',
                                outputCol='{final_name}', handleInvalid='skip')
                            features_stages.append({f['var']}_qtles) """))
                    else:
                        final_name = None
                if scaler and transform in ('keep', '', None):
                    # import pdb; pdb.set_trace()
                    old_final_name = final_name
                    final_name = final_name + '_scl'
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
                # if missing == 'constant' and transform != 'not_null':
                #     cte = f.get('constant')
                #     if cte is None:
                #         raise ValueError(gettext(
                #             "Missing constant value for attribute '{}'."
                #             ).format(f.get('name')))
                #     stmt = f"SELECT *, COALESCE({f['name']}, '{cte}') AS {f['var']}_na FROM __THIS__"
                #     code.append(dedent(f"""
                #         {f['var']}_na = feature.SQLTransformer(
                #             statement="{stmt}")
                #         features_stages.append({f['var']}_na) """))
                # elif missing == 'remove' and transform != 'not_null':
                #     stmt = f"SELECT * FROM __THIS__ WHERE ({f['name']}) IS NOT NULL"
                #     code.append(dedent(f"""
                #         {f['var']}_del_nulls = feature.SQLTransformer(
                #             statement="{stmt}")
                #         features_stages.append({f['var']}_del_nulls) """))
                #     f['na_name'] = f['name']

                if transform not in ('string_indexer', 'one_hot_encoder',
                                     'not_null', 'hashing'):
                    raise ValueError(gettext(
                        "Invalid transformation '{}' for attribute '{}'."
                    ).format(transform, f.get('name')))
                if transform == 'not_null':
                    final_name = final_name + '_na'
                    stmt = f"SELECT *, INT(ISNULL({f['var']})) AS {f['var']}_na FROM __THIS__"
                    code.append(dedent(f"""
                        {f['var']}_na = feature.SQLTransformer(
                            statement='{stmt}')
                        features_stages.append({f['var']}_na) """))

                else:  # transform in ('string_indexer', '', None):
                    code.append(dedent(f"""
                        {f['var']}_inx = feature.StringIndexer(
                            inputCol='{final_name}',
                            outputCol='{final_name}_inx',
                            handleInvalid='skip')
                        features_stages.append({f['var']}_inx) """))
                    final_name = final_name + '_inx'

                if transform == 'one_hot_encoder':
                    old_final_name = final_name
                    final_name = final_name + '_ohe'
                    code.append(dedent(f"""
                        {f['var']}_ohe = feature.OneHotEncoder(
                            inputCol='{old_final_name}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_ohe) """))
                self.features_names.append(final_name)
            elif data_type == 'textual':
                if transform == 'token_hash':
                    token_name = final_name + '_tkn'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.Tokenizer(
                            inputCol='{f['na_name']}',
                            outputCol='{token_name}')
                        features_stages.append({f['var']}_tkn) """))

                    final_name = token_name + '_hash'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.HashingTF(
                            inputCol='{token_name}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_tkn_hash) """))

                elif transform == 'token_stop_hash':
                    stop_name = final_name + '_stop'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.StopWordsRemover(
                            inputCol='{f['na_name']}',
                            outputCol='{stop_name}')
                        features_stages.append({f['var']}_stop) """))

                    token_name = stop_name + '_tkn'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.Tokenizer(
                            inputCol='{stop_name}',
                            outputCol='{token_name}')
                        features_stages.append({f['var']}_stop_tkn) """))

                    final_name = token_name + '_hash'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.HashingTF(
                            inputCol='{token_name}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_stop_tkn_hash) """))

                elif transform == 'count_vectorizer':
                    final_name = final_name + '_count_vectorizer'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.CountVectorizer(
                            inputCol='{f['na_name']}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_count_vectorizer) """))

                elif transform == 'word_2_vect':
                    final_name = final_name + '_word2vect'
                    code.append(dedent(f"""
                        {f['var']}_tkn = feature.Word2Vec(
                            inputCol='{f['na_name']}',
                            outputCol='{final_name}')
                        features_stages.append({f['var']}_word2vect) """))

            self.features_names.append(final_name)
            if f['usage'] == 'label':
                label = final_name

        supervisioned = self.task_type in ('regression', 'classification')
        if supervisioned:
            code.append(f"label = '{label}'")
            self.features_names.remove(label)

        return '\n'.join(code).strip()

    def get_final_features_names(self):
        return self.features_names or []

    def generate_code_for_missing_data_handling(self):
        """ Used by visualization builder only """
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
        code = ''
        if self.has_code:
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
        'cross_validation': 'CustomTrainValidationSplit',
    }

    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.strategy = parameters.get('strategy', 'split')
        self.seed = parameters.get('seed', 'None') or 'None'
        self.ratio = parameters.get('ratio', 0.8) or 0.8
        self.folds = parameters.get('folds', 10) or 10

    def generate_code(self):
        if self.strategy == 'split':
            code = dedent(f"""
            train_ratio = {self.ratio} # Between 0.01 and 0.99
            executor = CustomTrainValidationSplit(
                pipeline, evaluator, grid, train_ratio, seed={self.seed},
                strategy='{self.strategy}')
            """)
        elif self.strategy == 'cross_validation':
            code = dedent(f"""
            executor = CustomTrainValidationSplit(
                pipeline, evaluator, grid, seed={self.seed},
                strategy='{self.strategy}', folds={self.folds})
            """)

        return code.strip()


class GridOperation(ModelMetaOperation):
    """
    See:
    https://medium.com/storebrand-tech/random-search-in-spark-ml-5370dc908bd7
    https://towardsdatascience.com/hyperparameters-part-ii-random-search-on-spark-77667e68b606
    """

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
            pass
        elif self.strategy == 'random':
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
        self.types = _as_string_list(
            parameters.get('type', ['kmeans']),
            self.in_list('kmeans', 'bisecting'))

        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info,
                self.gt(1)),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'initMode': _as_string_list(parameters.get('init_mode'),
                                        self.in_list('random', 'k-means||')),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info, self.ge(0)),
            'distanceMeasure': _as_string_list(parameters.get('distance'),
                                               self.in_list('euclidean', 'cosine')),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
        }
        self.name = 'KMeans'
    '''
    def get_variations(self):
        result = []
        #import pdb; pdb.set_trace()
        #print(type(self.types))
        #print(self.types.value)
        if 'kmeans' in self.types:
            result.append(['KMeans', {}])
        if 'bisecting' in self.types:
            result.append(['BisectingKMeans', {'invalid': ['initMode']}])
        if len(result) == 0:
            result.append(['KMeans', {}])
        return result
    '''

    '''
    def get_variations(self):
        result = []
        types_values = types_values = self.types  # Acessando os valores do objeto HyperparameterInfo
        if 'kmeans' in types_values:
            result.append(['KMeans', {}])
        if 'bisecting' in types_values:
            result.append(['BisectingKMeans', {'invalid': ['initMode']}])
        if len(result) == 0:
            result.append(['KMeans', {}])
        return result
    '''

    def get_variations(self):
        result = []
        #import pdb; pdb.set_trace()
        #print(type(self.types))
        #print(self.types.value)
        if 'kmeans' in self.types.value:
            result.append(['KMeans', {}])
        if 'bisecting' in self.types.value:
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

class BisectingKMeansOperation(ClusteringOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClusteringOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = 'bisecting_kmeans'
        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info),
            'tol': _as_float_list(
                parameters.get('tolerance'), self.grid_info),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'minDivisibleClusterSize': _as_float_list(
                parameters.get('min_divisible_clusterSize'), self.grid_info),
            'distanceMeasure': _as_string_list(
                parameters.get('distance'), self.in_list('euclidean')),
        }
        self.name = 'BisectingKMeans'

class LDAOperation(ClusteringOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClusteringOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = 'lda'
        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info,
                self.gt(1)),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info, self.ge(0)),
            #'weightCol': _as_string_list(parameters.get('weight_col')),
            'featuresCol':_as_string_list(parameters.get('features')),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'checkpointInterval':_as_int_list(parameters.get('checkpoint_interval'), self.grid_info, self.ge(0)),
            'optimizer':_as_string_list(parameters.get('optimizer'),
                    self.in_list('online')),
            'learningOffset':_as_float_list(
                parameters.get('learning_offset'), self.grid_info),
            'learningDecay':_as_float_list(
                parameters.get('learning_decay'), self.grid_info),
            'subsamplingRate': _as_float_list(
                parameters.get('subsampling_rate'), self.grid_info),
            'optimizeDocConcentration':_as_boolean_list(
                parameters.get('optimize_doc_concentration')),
            'docConcentration':_as_float_list(parameters.get('doc_concentration'), self.grid_info),
            'topicConcentration':_as_float_list(parameters.get('topic_concentration'), self.grid_info),
            'topicDistributionCol':_as_string_list(parameters.get('topic_distribution_col')),
            'keepLastCheckpoint':_as_boolean_list(
                parameters.get('keep_last_checkpoint')),
        }
        self.name = 'LDA'

class PowerIterationClusteringOperation(ClusteringOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClusteringOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.var = 'pic'
        self.hyperparameters = {
            'k': _as_int_list(
                parameters.get('number_of_clusters'), self.grid_info,
                self.gt(1)),
            'initMode': _as_string_list(parameters.get('init_mode'),
                    self.in_list('random', 'degree')),
            'maxIter ': _as_int_list(
                parameters.get('max_iterations'), self.grid_info, self.ge(0)),
            #'weightCol': _as_string_list(parameters.get('weight_col')),
            'weightCol': parameters.get('weight'),
        }
        self.name = 'PIC'


class ClassificationOperation(EstimatorMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        EstimatorMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs, 'classification')


class DecisionTreeClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        self.hyperparameters = {
            'cacheNodeIds': _as_boolean_list(
                parameters.get('cache_node_ids')),
            'checkpointInterval':
                _as_int_list(parameters.get(
                    'checkpoint_interval'), self.grid_info,
                    self.ge(1)),
            'impurity': _as_string_list(parameters.get('impurity'),
                                        self.in_list('entropy', 'gini')),
            # 'leafCol': parameters.get('leaf_col'),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info,
                                    self.ge(2)),
            'maxDepth': _as_int_list(
                parameters.get('max_depth'), self.grid_info, self.ge(0)),
            # 'maxMemoryInMB': parameters.get('max_memory_in_m_b'),
            'minInfoGain': _as_float_list(
                parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
            _as_int_list(parameters.get(
                'min_instances_per_node'), self.grid_info, self.ge(1)),
            # 'minWeightFractionPerNode':
            #    parameters.get('min_weight_fraction_per_node'),None,
            #    self.between(0, 0.5, include_end=False)
            'seed': _as_int_list(parameters.get('seed'), None),
            # 'weightCol': parameters.get('weight_col')
        }
        self.var = 'decision_tree'
        self.name = 'DecisionTreeClassifier'


class GBTClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        params = {
            'cacheNodeIds': 'cache_node_ids',
            'checkpointInterval': 'checkpoint_interval',
            'lossType': 'loss_type',
            'maxBins': 'max_bins', 'maxDepth': 'max_depth',
            'maxIter': 'max_iter', 'minInfoGain': 'min_info_gain',
            'minInstancesPerNode': 'min_instances_per_node',
            'seed': 'seed', 'stepSize': 'step_size',
            'subsamplingRate': 'subsampling_rate'}

        self.hyperparameters = dict([
            (p, _as_list(parameters.get(v))) for p, v in params.items()])

        self.var = 'gbt_classifier'
        self.name = 'GBTClassifier'


class NaiveBayesClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'modelType': _as_string_list(
                parameters.get('model_type'),
                lambda x: [v for v in x if v not in
                           ['multinomial', 'bernoulli', 'gaussian']]),
            'smoothing': _as_float_list(
                parameters.get('smoothing'),
                self.grid_info,
                lambda x: [v for v in x if v < 0]
            ),
            # FIXME: implement in the interface
            # 'thresholds': _as_float_list(parameters.get('thresholds'),
            #                             self.grid_info),
            'weightCol': _as_string_list(parameters.get('weight_attribute')),
        }
        self.var = 'nb_classifier'
        self.name = 'NaiveBayes'


class PerceptronClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        layers = None
        '''
        if parameters.get('layers'):
            value = tuple(int(x.strip())
                          for x in parameters.get('layers').split(','))
            layers = HyperparameterInfo(
                value=(value,), param_type='list',
                values_count=1,
                random_generator='random_generator')
        '''
        if 'layers' in parameters and 'list' in parameters['layers']:
            layers = HyperparameterInfo(
            value=parameters['layers']['list'],  # Apenas a lista, sem a tupla externa
            param_type='list',
            values_count=1,
            random_generator='random_generator'
            )

        self.hyperparameters = {
            'layers': layers,
            'blockSize': _as_int_list(
                parameters.get('block_size'), self.grid_info),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'solver': _as_string_list(parameters.get('solver'),
                                      self.in_list('l-bfgs', 'gd'))
        }
        self.var = 'mlp_classifier'
        self.name = 'MultilayerPerceptronClassifier'



class RandomForestClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'impurity': _as_string_list(parameters.get('impurity')),
            'cacheNodeIds': _as_boolean_list(parameters.get('cache_node_ids')),
            'checkpointInterval':
                _as_int_list(parameters.get('checkpoint_interval')),
            'featureSubsetStrategy':
                _as_string_list(parameters.get('feature_subset_strategy')),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(
                parameters.get('max_depth'), self.grid_info),
            'minInfoGain': _as_float_list(
                parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode':
                _as_list(parameters.get('min_instances_per_node')),
            'numTrees': _as_int_list(
                parameters.get('num_trees'), self.grid_info),
            'seed': _as_list(parameters.get('seed')),
            'subsamplingRate': _as_list(parameters.get('subsampling_rate')),
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
            'fitIntercept': _as_boolean_list(
                parameters.get('fit_intercept')),
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

class FactorizationMachinesClassifierOperation(ClassificationOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ClassificationOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'factorSize':_as_int_list(parameters.get('factor_size'), self.grid_info),
            'fitLinear': _as_boolean_list(parameters.get('fit_linear')),
            'regParam': _as_float_list(parameters.get('reg_param'), self.grid_info), #int or float ??
            'miniBatchFraction': _as_float_list(parameters.get('min_batch'), self.grid_info),
            'initStd': _as_float_list(parameters.get('init_std'), self.grid_info),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'stepSize': _as_int_list(parameters.get('step_size'), self.grid_info),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'solver': _as_string_list(parameters.get('solver'),self.in_list('adamW', 'gd')),
            'seed': _as_int_list(parameters.get('seed'), None),
            'thresholds': _as_float_list(parameters.get('threshold'), None),
            #'weightCol': _as_string_list(parameters.get('weight_attr')),
        }
        self.var = 'fm_classifier'
        self.name = 'FMClassifier'


class RegressionOperation(EstimatorMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        EstimatorMetaOperation.__init__(
            self, parameters,  named_inputs,  named_outputs, 'regression')


class LinearRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)

        self.hyperparameters = {
            'aggregationDepth': # FIXME Missing in interface
            _as_int_list(parameters.get('aggregation_depth'), None,
                            self.ge(2)),
            'elasticNetParam':
                _as_float_list(parameters.get('elastic_net'), self.grid_info,
                            self.between(0, 1)),
            'epsilon':
                _as_float_list(parameters.get('epsilon'), self.grid_info,
                            self.gt(1)),
            'fitIntercept': _as_boolean_list(parameters.get('fit_intercept')),
            'loss': _as_string_list(parameters.get('loss'),
                                    self.in_list('huber', 'squaredError')),
            'maxIter': _as_int_list(
                parameters.get('max_iter'), self.grid_info,  self.ge(0)),
            'regParam': _as_float_list(parameters.get('reg_param'),
                                    self.grid_info, self.ge(0)),
            'solver': _as_string_list(parameters.get('solver'),
                                    self.in_list('auto', 'normal', 'l-bfgs')),
            'standardization': _as_boolean_list(parameters.get('standardization')),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'weightCol': parameters.get('weight'), # Missing in interface
        }
        self.var = 'linear_reg'
        self.name = 'LinearRegression'

    def get_constrained_params(self):
        result = []
        # result.append(
        #        f'{{{self.var}.epsilon: {repr(family)}, '
        #        f'{self.var}.loss: "huber"')
        return result


class IsotonicRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'isotonic': _as_boolean_list(parameters.get('isotonic')),
            'weightCol': _as_string_list(parameters.get('weight_col')),
        }
        self.var = 'isotonic_reg'
        self.name = 'IsotonicRegression'


class GBTRegressorOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'cacheNodeIds': _as_boolean_list(parameters.get('cache_node_ids')),
            'checkpointInterval': _as_int_list(parameters.get('checkpoint_interval')),
            'featureSubsetStrategy': _as_string_list(parameters.get('feature_subset_strategy')),
            'impurity': _as_string_list(parameters.get('impurity')),
            'leafCol': _as_string_list(parameters.get('leaf_col')),
            'lossType': _as_string_list(parameters.get('loss_type')),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(parameters.get('max_depth'), self.grid_info),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'maxMemoryInMB': parameters.get('max_memory_in_m_b'),
            'minInfoGain':  _as_float_list(parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode': _as_int_list(parameters.get('min_instance'), self.grid_info),
            'minWeightFractionPerNode':
                parameters.get('min_weight_fraction_per_node'),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'stepSize': _as_int_list(parameters.get('step_size'), self.grid_info),
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
            'minInstancesPerNode':  _as_int_list(parameters.get('min_instances_per_node'), self.grid_info),
        }
        self.var = 'dt_reg'
        self.name = 'DecisionTreeRegressor'


class RandomForestRegressorOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'bootstrap': _as_boolean_list(parameters.get('bootstrap')),
            'cacheNodeIds':_as_boolean_list(parameters.get('cache_node_ids')),
            'checkpointInterval':_as_int_list(parameters.get('checkpoint_interval'), self.grid_info),
            'featureSubsetStrategy':_as_string_list(parameters.get('feature_subset_strategy')),
            'impurity': _as_string_list(parameters.get('impurity')),
            'leafCol': _as_string_list(parameters.get('leaf_col')),
            'maxBins': _as_int_list(parameters.get('max_bins'), self.grid_info),
            'maxDepth': _as_int_list(parameters.get('max_depth'), self.grid_info),
            'maxMemoryInMB': _as_int_list(parameters.get('max_memory_in_m_b'), self.grid_info),
            'minInfoGain': _as_float_list(parameters.get('min_info_gain'), self.grid_info),
            'minInstancesPerNode': _as_list(parameters.get('min_instances_per_node')),
            'minWeightFractionPerNode': _as_float_list(parameters.get('min_weight_fraction_per_node'), self.grid_info),
            'numTrees': _as_int_list(parameters.get('num_trees'), self.grid_info),
            'seed': _as_int_list(parameters.get('seed'), self.grid_info),
            'subsamplingRate': _as_float_list(parameters.get('subsampling_rate'), self.grid_info),
            'weightCol': _as_string_list(parameters.get('weight_col'))
        }
        self.var = 'rand_forest_reg'
        self.name = 'RandomForestRegressor'


class GeneralizedLinearRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.family_link = parameters.get('family_link') or []
        if 'solver' in parameters:
            parameters['solver'] = dict((s, v) for s, v in parameters['solver'].items()
                                    if s != 'auto')

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
        for family, link in [x.split(':')
                for x in self.family_link.get('list', [])]:
            result.append(
                f'{{{self.var}.family: {repr(family)}, '
                f'{self.var}.link: {repr(link)}}}')
        return result

class FactorizationMachinesRegressionOperation(RegressionOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        RegressionOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.hyperparameters = {
            'factorSize':_as_int_list(parameters.get('factor_size'), self.grid_info),
            'fitLinear': _as_boolean_list(parameters.get('fit_linear')),
            'regParam': _as_float_list(parameters.get('reg_param'), self.grid_info), #int or float ??
            'miniBatchFraction': _as_float_list(parameters.get('min_batch'), self.grid_info),
            'initStd': _as_float_list(parameters.get('init_std'), self.grid_info),
            'maxIter': _as_int_list(parameters.get('max_iter'), self.grid_info),
            'stepSize': _as_int_list(parameters.get('step_size'), self.grid_info),
            'tol': _as_float_list(parameters.get('tolerance'), self.grid_info),
            'solver': _as_string_list(parameters.get('solver'),self.in_list('adamW', 'gd')),
            'seed': _as_int_list(parameters.get('seed'), None),
            #'weightCol': _as_string_list(parameters.get('weight_attr')),
            #'stringIndexerOrderType': _as_string_list(parameters.get('stringIndexerOrderType'),self.in_list('frequencyDesc', 'frequencyAsc', 'alphabetDesc',
            #'alphabetAsc')),

        }

        self.var = 'fm_reg'
        self.name = 'FMRegressor'

class VisualizationOperation(MetaPlatformOperation):

    DEFAULT_PALETTE = ['#636EFA', '#EF553B',
                       '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
                       '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    CHART_MAP_TYPES = ('scattermapbox', )

    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(
            self, parameters,  named_inputs,  named_outputs)
        self.type = self.get_required_parameter(parameters, 'type')
        self.palette = parameters.get('palette')
        self.display_legend = parameters.get('display_legend', 'HIDE')
        if self.type not in self.CHART_MAP_TYPES:
            self.x = self.get_required_parameter(parameters, 'x')
            self.y = self.get_required_parameter(parameters, 'y')
            self.x_axis = self.get_required_parameter(parameters, 'x_axis')
            self.y_axis = self.get_required_parameter(parameters, 'y_axis')
        else:
            self.x = None
            self.y = None
            self.x_axis = None
            self.y_axis = None
            self.latitude = self.get_required_parameter(parameters, 'latitude')
            self.longitude = self.get_required_parameter(
                parameters, 'longitude')

        self.parameters = parameters

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
            k: {'value': getattr(self, k)} for k in
            ['type', 'display_legend', 'palette', 'x', 'y', 'x_axis', 'y_axis']
        })
        task_obj['forms']['y']['value'] = [
            y for y in task_obj['forms']['y']['value']
            if y.get('enabled')
        ]
        if len(task_obj['forms']['y']['value']) == 0:
            raise ValueError(gettext('There is no series or none is enabled'))
        for p in ['hole', 'text_position', 'text_info', 'smoothing', 'color_scale',
                  'auto_margin', 'right_margin', 'left_margin', 'top_margin', 'bottom_margin',
                  'title', 'template', 'blackWhite', 'subgraph', 'subgraph_orientation',
                  'animation', 'height', 'width', 'opacity', 'fill_opacity',
                  'color_attribute', 'size_attribute', 'number_format', 'text_attribute',
                  'style', 'tooltip_info', 'zoom', 'center_latitude', 'center_longitude',
                  'marker_size', 'limit', 'filter']:
            task_obj['forms'][p] = {'value': self.parameters.get(p)}

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


