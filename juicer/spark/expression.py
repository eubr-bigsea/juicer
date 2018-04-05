import functools
from textwrap import dedent

from juicer.util import group


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.build_functions_dict()
        self.parsed_expression = self.parse(json_code, params)

    def parse(self, tree, params):
        if tree['type'] == 'BinaryExpression':
            result = "({} {} {})".format(
                self.parse(tree['left'], params), tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            result = "'{}'".format(v) if type(v) in [str, unicode] else str(v)

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
                "!": "~"
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
        else:
            raise ValueError(_("Unknown type: {}").format(tree['type']))
        return result

    def get_window_function(self, spec, params):
        """
        Window function is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        bins_size = '{} seconds'.format(arguments[-2])

        # FIXME: add the word 'SECONDS' after parameter 'SEGUNDOS'
        result = (
            "functions.window({value}, '{bin}')"
            ".{start_or_end}.cast('timestamp')").format(
            value=', '.join(arguments[:-2]), bin=bins_size,
            start_or_end=field_name)
        return result

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

    def get_strip_accents_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        strip_accents = (
            "functions.udf("
            "lambda text: ''.join(c for c in unicodedata.normalize('NFD', text) "
            "if unicodedata.category(c) != 'Mn'), "
            "types.StringType())"
        )

        result = '{}({})'.format(strip_accents, arguments)

        return result

    def get_strip_punctuation_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])

        strip_punctuation = (
            "functions.udf("
            "lambda text: text.translate("
            "dict((ord(char), None) for char in string.punctuation)), "
            "types.StringType())"
        )

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

    def build_functions_dict(self):
        spark_sql_functions = {
            'add_months': self.get_function_call,
            'ceil': self.get_function_call,
            'coalesce': self.get_function_call,
            'col': self.get_function_call,
            'concat': self.get_function_call,
            'concat_ws': self.get_function_call,
            'current_date': self.get_function_call,
            'current_timestamp': self.get_function_call,
            'date_add': self.get_function_call,
            'date_format': self.get_function_call,
            'date_sub': self.get_function_call,
            'datediff': self.get_function_call,
            'dayofmonth': self.get_function_call,
            'dayofyear': self.get_function_call,
            'exp': self.get_function_call,
            'expr': self.get_function_call,
            'floor': self.get_function_call,
            'format_number': self.get_function_call,
            'format_string': self.get_function_call,
            'from_json': self.get_function_call,
            'from_unixtime': self.get_function_call,
            'from_utc_timestamp': self.get_function_call,
            'greatest': self.get_function_call,
            'hour': self.get_function_call,
            'instr': self.get_function_call,
            'isnan': self.get_function_call,
            'isnull': self.get_function_call,
            'last_day': self.get_function_call,
            'least': self.get_function_call,
            'length': self.get_function_call,
            'levenshtein': self.get_function_call,
            'lit': self.get_function_call,
            'locate': self.get_function_call,
            'log': self.get_function_call,
            'log10': self.get_function_call,
            'log2': self.get_function_call,
            'lower': self.get_function_call,
            'lpad': self.get_function_call,
            'ltrim': self.get_function_call,
            'md5': self.get_function_call,
            'minute': self.get_function_call,
            'monotonically_increasing_id': self.get_function_call,
            'month': self.get_function_call,
            'months_between': self.get_function_call,
            'next_day': self.get_function_call,
            'pow': self.get_function_call,
            'rand': self.get_function_call,
            'randn': self.get_function_call,
            'regexp_extract': self.get_function_call,
            'regexp_replace': self.get_function_call,
            'repeat': self.get_function_call,
            'reverse': self.get_function_call,
            'round': self.get_function_call,
            'rpad': self.get_function_call,
            'rtrim': self.get_function_call,
            'second': self.get_function_call,
            'split': self.get_function_call,
            'sqrt': self.get_function_call,
            'substring': self.get_function_call,
            'substring_index': self.get_function_call,
            'toDegrees': self.get_function_call,
            'toRadians': self.get_function_call,
            'to_date': self.get_function_call,
            'to_json': self.get_function_call,
            'to_utc_timestamp': self.get_function_call,
            'translate': self.get_function_call,
            'trim': self.get_function_call,
            'trunc': self.get_function_call,
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
            'group_datetime': self.get_window_function,
            'set_of_ints': functools.partial(self.get_set_function,
                                             data_type='IntegerType'),
            'set_of_strings': functools.partial(self.get_set_function,
                                                data_type='StringType'),
            'set_of_floats': functools.partial(self.get_set_function,
                                               data_type='FloatType'),
            'strip_accents': self.get_strip_accents_function,
            'strip_punctuation': self.get_strip_punctuation_function,
            'when': self.get_when_function,
            'window': self.get_window_function,
        }

        self.functions.update(spark_sql_functions)
        self.functions.update(custom_functions)
