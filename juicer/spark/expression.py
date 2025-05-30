# coding=utf-8

import functools
import json
from textwrap import dedent
from typing import Callable

from juicer.util import group
from six import text_type

# https://github.com/microsoft/pylance-release/issues/140#issuecomment-661487878
_: Callable[[str], str] 

class Expression(object):
    def __init__(self, json_code, params, window=False):
        self.window = window
        self.code = json_code
        self.functions = {}
        self.add_import_udf = None
        self.register_udf_mapping = {}
        self.build_functions_dict()
        self.parsed_expression = self.parse(json_code, params)

    def parse(self, tree, params):
        if tree['type'] == 'BinaryExpression':
            op = tree['operator']
            if op == '===':
                op = '=='
            elif op == '!==':
                op = '!='
            result = "({} {} {})".format(
                self.parse(tree['left'], params), tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            if isinstance(v, (str, text_type)):
                result = "'{}'".format(v)
            elif v is None:
                result = None
            else:
                result = str(v)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            result = self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                result = "{}['{}']".format(params['input'], tree['name'])
            else:
                result = "functions.col('{}')".format(tree['name'])

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = '~'
            result = "({} {})".format(tree['operator'],
                                      self.parse(tree['argument'], params))

        elif tree['type'] == 'LogicalExpression':
            operators = {
                "&&": "&",
                "||": "|",
                "!": "~",
            }
            operator = operators[tree['operator']]
            result = "({}) {} ({})".format(self.parse(tree['left'], params),
                                           operator,
                                           self.parse(tree['right'], params))
        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [
                tree['test'], tree['consequent'], tree['alternate']
            ]}
            result = self.get_when_function(spec, params)
        elif tree['type'] == 'Compound':
            raise ValueError(
                _("Transformation has an invalid expression. "
                  "Maybe variables are using spaces in their names."))
        elif tree['type'] == 'ArrayExpression':
            array = []
            for elem in tree['elements']:
                if elem['type'] == 'Literal':
                    array.append(elem['value'])
                else:
                    array.append(self.parse(elem, params))
            result = array
        else:
            raise ValueError(_("Unknown type: {}").format(tree['type']))
        return result

    def get_time_window_function(self, spec, params):
        """
        Window function is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        bins_size = '{} seconds'.format(arguments[len(arguments) - 2])

        # FIXME: add the word 'SECONDS' after parameter 'SEGUNDOS'
        result = (
            "functions.window({value}, str('{bin}'))"
            ".{start_or_end}.cast('timestamp')").format(
            value=', '.join(arguments[: 1 - len(arguments)]), bin=bins_size,
            start_or_end=field_name)
        return result

    def get_column_function(self, spec, params, arg_count=0):
        """
        Functions that are not available at pyspark.sql.functions, but are
        implemented in pyspark.sql.Column class
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        if len(arguments) != arg_count + 1:
            raise ValueError(_("Invalid parameters for function: {}()").format(
                spec['callee']['name']))

        return '{col}.{f}({args})'.format(
            col=arguments[0], f=spec['callee']['name'],
            args=', '.join(arguments[1:]))

    def get_set_function(self, spec, params, data_type):
        """
        Removes duplicates from a list or vector column using set collection in
        Python.
        """
        # Evaluates if column name is wrapped in a col() function call
        arguments = [self.parse(x, params) for x in spec['arguments']]

        if len(arguments) != 1:
            raise ValueError(_('Function set() expects exactly 1 argument.'))

        get_set_function = dedent("""
            functions.udf(lambda values: [v for v in set(values) if v != ''],
                        types.ArrayType(types.{type}()))
        """).format(type=data_type).strip()

        return '{}({})'.format(get_set_function, arguments[0])

    def get_when_function(self, spec, params):
        """
        Map when() function call in Lemonade into when() call in Spark.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        # print >> sys.stderr, group(arguments[:-1], 2)
        code = ["functions.when({}, {})".format(cond, value) for cond, value in
                group(arguments[:-1], 2)]
        if arguments[-1] is not None:
            code.append("otherwise({})".format(arguments[-1]))

        return '.'.join(code)

    def get_ith_function(self, spec, params):
        """
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        f = 'juicer_ext.ith_function_udf'
        result = '{}({}, functions.lit({}))'.format(f, arguments[0],
                                                    arguments[1])
        return result

    def get_strip_accents_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        # strip_accents = (
        #     "functions.udf("
        #     "lambda text: ''.join(c for c in unicodedata.normalize('NFD', text) "
        #     "if unicodedata.category(c) != 'Mn'), "
        #     "types.StringType())"
        # )
        strip_accents = 'juicer_ext.strip_accents_udf'

        result = '{}({})'.format(strip_accents, arguments)

        return result

    def get_strip_punctuation_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        # strip_punctuation = (
        #     "functions.udf("
        #     "lambda text: text.translate("
        #     "dict((ord(char), None) for char in string.punctuation)), "
        #     "types.StringType())"
        # )
        strip_punctuation = 'juicer_ext.remove_punctuation_udf'

        result = '{}({})'.format(strip_punctuation, arguments)

        return result

    def get_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        # callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        result = 'functions.{}({})'.format(spec['callee']['name'],
                                           arguments)
        return result

    def get_function_call_with_columns(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        # callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = []
        for x in spec['arguments']:
            if x['type'] == 'Literal':
                v = 'functions.lit({})'.format(self.parse(x, params))
            else:
                v = self.parse(x, params)
            arguments.append(v)

        result = 'functions.{}({})'.format(spec['callee']['name'],
                                           ', '.join(arguments))
        return result

    def get_log_call(self, spec, params):
        """
        Handle log function, converting the base (if informed) to double type.
        JS library used to parse expressions converts double with no value
        after separator to int (e.g. 2.0 = 2) and this causes an error in Spark.
        """
        if len(spec) == 2:
            spec[0] = str(float(spec[0]))

        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        result = 'functions.{}({})'.format(spec['callee']['name'],
                                           arguments)
        return result

    def get_window_function(self, spec, params):
        """ """

        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        result = 'functions.{}({}).over({})'.format(
            spec['callee']['name'], arguments, params.get('window', 'window'))
        return result

    def get_translate_function(self, spec, params):
        """
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        f = 'juicer_ext.translate_function_udf'
        result = dedent('''{}(
                    {}, {}, {},
                    {})'''.format(f, arguments[0], arguments[1], arguments[2],
                                  json.dumps(arguments[3])))
        return result

    def get_expr_function(self, spec, params):
        """
        Generate code for Expr spark function (that can be used to access Java Spark UDF)
        """
        arguments = [t['name'] for t in spec['arguments']]
        function_name = spec['callee']['name']
        result = 'functions.expr("{}({})")'.format(
            function_name, ', '.join(arguments))

        if function_name in self.register_udf_mapping:
            self.add_import_udf = self.register_udf_mapping[function_name]
        
        return result

    def build_functions_dict(self):
        spark_sql_functions = {
            'abs': self.get_function_call,
            'acos': self.get_function_call,
            'add_months': self.get_function_call,
            'array': self.get_function_call,
            'array_contains': self.get_function_call,
            'array_distinct': self.get_function_call,
            'array_except': self.get_function_call,
            'array_intersect': self.get_function_call,
            'array_join': self.get_function_call,
            'array_max': self.get_function_call,
            'array_min': self.get_function_call,
            'array_position': self.get_function_call,
            'array_remove': self.get_function_call,
            'array_repeat': self.get_function_call,
            'array_sort': self.get_function_call,
            'array_union': self.get_function_call,
            'arrays_overlap': self.get_function_call,
            'arrays_zip': self.get_function_call,
            'ascii': self.get_function_call,
            'asin': self.get_function_call,
            'atan': self.get_function_call,
            'atan2': self.get_function_call,
            'base64': self.get_function_call,
            'basestring': self.get_function_call,
            'bin': self.get_function_call,
            'bitwiseNOT': self.get_function_call,
            'bround': self.get_function_call,
            'cbrt': self.get_function_call,
            'ceil': self.get_function_call,
            'coalesce': self.get_function_call,
            'col': self.get_function_call,
            'concat': self.get_function_call_with_columns,
            'concat_ws': self.get_function_call,
            'conv': self.get_function_call,
            'cos': self.get_function_call,
            'cosh': self.get_function_call,
            'crc32': self.get_function_call,
            'create_map': self.get_function_call,
            'current_date': self.get_function_call,
            'current_timestamp': self.get_function_call,
            'date_add': self.get_function_call,
            'date_format': self.get_function_call,
            'date_sub': self.get_function_call,
            'date_trunc': self.get_function_call,
            'datediff': self.get_function_call,
            'dayofmonth': self.get_function_call,
            'dateofweek': self.get_function_call,
            'dayofyear': self.get_function_call,
            'decode': self.get_function_call,
            'degrees': self.get_function_call,
            'element_at': self.get_function_call,
            'encode': self.get_function_call,
            'exp': self.get_function_call,
            'explode': self.get_function_call,
            'explode_outer': self.get_function_call,
            'expm1': self.get_function_call,
            'expr': self.get_function_call,
            'flatten': self.get_function_call,
            'floor': self.get_function_call,
            'format_number': self.get_function_call,
            'from_json': self.get_function_call,
            'format_string': self.get_function_call,
            'from_unixtime': self.get_function_call,
            'from_utc_timestamp': self.get_function_call,
            'get_json_object': self.get_function_call,
            'greatest': self.get_function_call,
            'hash': self.get_function_call,
            'hex': self.get_function_call,
            'hour': self.get_function_call,
            'hypot': self.get_function_call,
            'initcap': self.get_function_call,
            'instr': self.get_function_call,
            'isnan': self.get_function_call,
            'isnull': self.get_function_call,
            'json_tuple': self.get_function_call,
            'last_day': self.get_function_call,
            'least': self.get_function_call,
            'length': self.get_function_call,
            'levenshtein': self.get_function_call,
            'lit': self.get_function_call,
            'locate': self.get_function_call,
            'log': self.get_log_call,
            'log10': self.get_function_call,
            'log1p': self.get_function_call,
            'log2': self.get_function_call,
            'lower': self.get_function_call,
            'lpad': self.get_function_call,
            'ltrim': self.get_function_call,
            'map_concat': self.get_function_call,
            'map_keys': self.get_function_call,
            'map_values': self.get_function_call,
            'md5': self.get_function_call,
            'minute': self.get_function_call,
            'monotonically_increasing_id': self.get_function_call,
            'month': self.get_function_call,
            'months_between': self.get_function_call,
            'nanvl': self.get_function_call,
            'next_day': self.get_function_call,
            'posexplode': self.get_function_call,
            'pow': self.get_function_call,
            'quarter': self.get_function_call,
            'radians': self.get_function_call,
            'rand': self.get_function_call,
            'randn': self.get_function_call,
            'regexp_extract': self.get_function_call,
            'regexp_replace': self.get_function_call,
            'repeat': self.get_function_call,
            'reverse': self.get_function_call,
            'rint': self.get_function_call,
            'round': self.get_function_call,
            'rpad': self.get_function_call,
            'rtrim': self.get_function_call,
            'schema_of_json': self.get_function_call,
            'second': self.get_function_call,
            'sequence': self.get_function_call,
            'sha1': self.get_function_call,
            'sha2': self.get_function_call,
            'shiftLeft': self.get_function_call,
            'shiftRight': self.get_function_call,
            'shiftRightUnsigned': self.get_function_call,
            'shuffle': self.get_function_call,
            'signum': self.get_function_call,
            'sin': self.get_function_call,
            'sinh': self.get_function_call,
            'size': self.get_function_call,
            'slice': self.get_function_call,
            'sort_array': self.get_function_call,
            'soundex': self.get_function_call,
            'split': self.get_function_call,
            'sqrt': self.get_function_call,
            'struct': self.get_function_call,
            'substring': self.get_function_call,
            'substring_index': self.get_function_call,
            'tan': self.get_function_call,
            'tanh': self.get_function_call,
            'to_date': self.get_function_call,
            'to_json': self.get_function_call,
            'to_timestamp': self.get_function_call,
            'to_utc_timestamp': self.get_function_call,
            'translate': self.get_function_call,
            'toDegrees': self.get_function_call,
            'toRadians': self.get_function_call,
            'translate': self.get_function_call,
            'trim': self.get_function_call,
            'trunc': self.get_function_call,
            'unbase64': self.get_function_call,
            'unhex': self.get_function_call,
            'unix_timestamp': self.get_function_call,
            'upper': self.get_function_call,
            'weekofyear': self.get_function_call,
            'year': self.get_function_call,
        }

        # Functions that does not exist on Spark, but can be implemented in
        # Python as a UDF. For now, and due simplicity, we require that every
        # custom function is necessarily defined here. Also, we
        # should not use 'get_function_call' for code generation in this case.
        custom_functions = {
            'group_datetime': self.get_time_window_function,
            'set_of_ints': functools.partial(self.get_set_function,
                                             data_type='IntegerType'),
            'set_of_strings': functools.partial(self.get_set_function,
                                                data_type='StringType'),
            'set_of_floats': functools.partial(self.get_set_function,
                                               data_type='FloatType'),
            'strip_accents2': self.get_strip_accents_function,
            'strip_punctuation': self.get_strip_punctuation_function,
            'when': self.get_when_function,
            'window': self.get_time_window_function,
            'ith': self.get_ith_function,
            # 'translate': self.get_translate_function,
            "validate_codes": self.get_expr_function,
            "date_patterning": self.get_expr_function,
            "physical_or_legal_person": self.get_expr_function,
            "strip_accents": self.get_expr_function,
            "complete_cpf_cnpj": self.get_expr_function,
        }

        column_functions = {
            'alias': functools.partial(self.get_column_function, arg_count=1),
            'astype': functools.partial(self.get_column_function, arg_count=1),
            'between': functools.partial(self.get_column_function, arg_count=2),
            'cast': functools.partial(self.get_column_function, arg_count=1),
            'contains': functools.partial(self.get_column_function,
                                          arg_count=1),
            'endswith': functools.partial(self.get_column_function,
                                          arg_count=1),
            'isNotNull': functools.partial(self.get_column_function),
            'isNull': functools.partial(self.get_column_function),
            'like': functools.partial(self.get_column_function, arg_count=1),
            'rlike': functools.partial(self.get_column_function, arg_count=1),
            'startswith': functools.partial(self.get_column_function,
                                            arg_count=1),
            'substr': functools.partial(self.get_column_function, arg_count=1),

        }

        self.functions.update(spark_sql_functions)
        self.functions.update(custom_functions)
        self.functions.update(column_functions)

        if self.window:
            window_functions = {
                'max': self.get_window_function,
                'min': self.get_window_function,
                'count': self.get_window_function,
                'avg': self.get_window_function,
                'sum': self.get_window_function,
                'std': self.get_window_function,
                'lag': self.get_window_function,
                'lead': self.get_window_function,
                'first': self.get_window_function,
                'last': self.get_window_function,
                'ntile': self.get_window_function,
                'cume_dist': self.get_window_function,
                'percent_rank': self.get_window_function,
                'rank': self.get_window_function,
                'dense_rank': self.get_window_function,
                'row_number': self.get_window_function,
            }
            self.functions.update(window_functions)

        self.register_udf_mapping = {
            "validate_codes": 'spark_session.udf.registerJavaFunction("validate_codes", "br.ufmg.dcc.lemonade.udfs.ValidateCodesUDF", types.BooleanType())',
            "date_patterning": 'spark_session.udf.registerJavaFunction("date_patterning", "br.ufmg.dcc.lemonade.udfs.DatePatterningUDF", types.StringType())',
            "physical_or_legal_person": 'spark_session.udf.registerJavaFunction("physical_or_legal_person", "br.ufmg.dcc.lemonade.udfs.PhysicalOrLegalPersonUDF", types.StringType())',
            "strip_accents": 'spark_session.udf.registerJavaFunction("strip_accents", "br.ufmg.dcc.lemonade.udfs.StripAccentsUDF", types.StringType())',
            "complete_cpf_cnpj": 'spark_session.udf.registerJavaFunction("complete_cpf_cnpj", "br.ufmg.dcc.lemonade.udfs.CompleteCpfCnpjUDF", types.StringType())',
        }
