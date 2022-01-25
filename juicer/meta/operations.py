import json
import re
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
        self.target_platform = 'scikit-learn'

    def get_required_parameter(self, parameters, name):
        if name not in parameters:
            raise ValueError(gettext('Missing required parameter: {}').format(
                name))
        else:
            return parameters.get(name)

    def generate_flows(self, next_task):
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
        return {
            "id": self.new_id,
            "display_order": self.task.get('display_order', 0),
            "environment": "DESIGN",
            "forms": {
              "display_schema": {"value": "1"},
              "display_sample": {"value": f"{self.last}"},
              "sample_size": {"value": self.parameters[
                'transpiler'].sample_size},
              "display_text": {"value": "1"}
            },
            "name": self.task['name'],
            "enabled": self.task['enabled'],
            "left": (order % 4)* 250 + 100,
            "top": (order // 4) * 150 + 100,
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
    array_index_re = re.compile(r'(?:\D*?)(-?\d+)(?:\D?)')
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

            'invert-boolean': {'f': None, 'op': '!'},
            'extract-from-array': {'f': None, 'op': ''},
        }
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        op = parameters.get('task').get('operation')
        self.slug = op.get('slug')

        self.attributes = self.get_required_parameter(parameters, 'attributes')

    def generate_code(self):
        task_obj = self._get_task_obj()

        info = self.SLUG_TO_EXPR[self.slug]
        function_name = info.get('f')
        expressions = []
        if function_name:

            self.form_parameters = {}
            for arg in info.get('args', []):
                if arg[0] == '{' and arg[-1] == '}':
                    self.form_parameters[arg[1:-1]] = self.parameters.get(
                        arg[1:-1])
            # import sys
            #print(self.form_parameters, file=sys.stderr)

            # Convert the parameters
            function_args = [arg.format(**self.form_parameters) for arg in
                info.get('args', [])]

            final_args_str = ''
            final_args = []

            if function_args:
                final_args_str = ', ' + ', '.join(function_args)
                transform = info['transform']
                for i, arg in enumerate(function_args):
                    v = transform[i](arg)
                    final_args.append(
                    {'type': 'Literal', 'value': v, 'raw': f'{v}'})
            # Uses the same attribute name as alias, so it will be overwritten
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
                  })
        elif self.slug == 'invert-boolean':
            for attr in self.attributes:
                expressions.append(
                  {
                    'alias': attr,
                    'expression': f'!{attr}',
                    'tree': {
                       'type': 'UnaryExpression', 'operator': '!',
                        'argument': {'type': 'Identifier',  'name': attr},
                        'prefix': True
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
                        'arguments':[
                            {'type': 'Identifier',  'name': attr},
                            {'type': 'Literal',  'value': index, 'raw': f'"{index}"' }
                         ],
                        'callee': {'type': 'Identifier', 'name': 'element_at'},
                    }
                  })

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

class CastOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.cast_attributes = self.get_required_parameter(parameters, 'cast_attributes')
        self.errors = self.get_required_parameter(parameters, 'errors')
        self.invalid_values = parameters.get('invalid_values')

    def generate_code(self):
        task_obj = self._get_task_obj()
        for prop in ['cast_attributes', 'errors', 'invalid_values']:
            value = getattr(self, prop)
            task_obj['forms'][prop] = {'value': value}
        task_obj['operation'] = {"id": 140}
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
        
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        if isinstance(self.attributes, list):
            self.attributes = self.attributes[0]

        self.find = self.get_required_parameter(parameters, 'find')
        self.replace= self.get_required_parameter(parameters, 'replace')

    def generate_code(self):
        task_obj = self._get_task_obj()
        attr = self.attributes
        formula = {
             'alias': attr,
             'expression': f'when({attr} == {self.find}, {self.replace}, {attr})',
             'tree': {
                 'type': 'CallExpression',
                 'arguments': [
                    {'type': 'BinaryExpression', 'operator': '==',
                     'left': {'type': 'Identifier', 'name': attr},
                     'right': {'type': 'Literal', 'value': self.find, 
                        'raw': f'{self.find}'}
                    },
                    {'type': 'Literal', 'value': self.replace, 
                        'raw': f'{self.replace}'},
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

class SelectOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.mode = parameters.get('mode', 'include') or 'include'
        self.output_port_name = 'output projected data'

    def generate_code(self):
        task_obj = self._get_task_obj()
        task_obj['forms'].update({
          "attributes": {"value": self.attributes},
          "mode": {"value": self.mode},
        })
        task_obj['operation'] = {"id": 6}
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
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.attributes = self.get_required_parameter(parameters, 'attributes')
        self.indexes = exp_index.findall(parameters.get('indexes', '') or '')

class SaveOperation(MetaPlatformOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
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
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.data_source= self.get_required_parameter(parameters, 'data_source')
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
        MetaPlatformOperation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.data_source= self.get_required_parameter(parameters, 'data_source')
        
        self.task_id = self.task.get('id')
        self.other_id = f'{self.task_id}-1'
        
        self.input_port_name = 'input data 1'
        self.parameter_names = ['keep_right_keys', 'match_case', 'join_parameters']

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

class ModelMetaOperation(Operation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task = parameters.get('task')

        self.has_code = True

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
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.task_type = parameters.get('task_type', 'binary-classification')


    def generate_code(self):
        metric = self.TYPE_TO_METRIC_PARAM[self.task_type]
        evaluator = self.TYPE_TO_CLASS[self.task_type]
        code = dedent(
            f"""
            # Pipeline evaluator
            evaluator = evaluation.{evaluator}(labelCol=label, metricName='{metric}')
            stages.append(evaluator)
            """)
        return code.strip()

class FeaturesOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class FeaturesReductionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.method = parameters.get('method', 'pca') or 'pca'
        self.k = int(parameters.get('k', 2) or 2)

    def generate_code(self):
        code = dedent(f"""
            # Feature reduction
            feature_reducer = feature.PCA(k={self.k})
            stages.append(feature_reducer)
        """)
        return code.strip()

class SplitOperation(ModelMetaOperation):
    STRATEGY_TO_CLASS = {
        'split': 'CustomTrainValidationSplit',
        'cross_validation': 'CustomCrossValidation',
    }
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.strategy = parameters.get('strategy', 'split')
        self.seed = parameters.get('seed', 'None') or 'None'
        self.ratio = parameters.get('ratio', 0.8) or 0.8

    def generate_code(self):
        if self.strategy == 'split':
            code = dedent(f"""
            seed = {self.seed} # random number generation
            train_ratio = {self.ratio} # Between 0.01 and 0.99
            split_strategy = CustomTrainValidationSplit(
                pipeline, evaluator, train_ratio, seed)
            """)
        elif self.strategy == 'cross_validation':
            code = dedent(f"""
            """)

        return code.strip()

class GridOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)


class KMeansOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class GaussianMixOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class DecisionTreeClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class GBTClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class NaiveBayesClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class PerceptronClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class RandomForestClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class LogisticRegressionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class SVMClassifierOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class LinearRegressionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class IsotonicRegressionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class GBTRegressorOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class RandomForestRegressorOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

class GeneralizedLinearRegressionOperation(ModelMetaOperation):
    def __init__(self, parameters,  named_inputs, named_outputs):
        ModelMetaOperation.__init__(self, parameters,  named_inputs,  named_outputs)

