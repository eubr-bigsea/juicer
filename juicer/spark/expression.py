import sys

from juicer.util import group


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.build_functions_dict()
        self.parsed_expression = self.parse(json_code, params)

    def parse(self, tree, params):
        if tree['type'] == 'BinaryExpression':
            result = "{} {} {}".format(
                self.parse(tree['left'], params), tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            result = "'{}'".format(v) if type(v) in [str, unicode] else str(v)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            result = self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                result = "{}.{}".format(params['input'], tree['name'])
            else:
                result = "col('{}')".format(tree['name'])

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
            raise ValueError("Unknown type: {}".format(tree['type']))
        return result

    def get_window_function(self, spec, params):
        """
        Window funciton is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        bins_size = '{} seconds'.format(arguments[-2])

        # FIXME: add the workd 'SECONDS' after parameter 'SEGUNDOS'
        result = (
            "functions.window(functions.from_unixtime({value}/1e6), '{bin}')"
            ".{start_or_end}.cast('timestamp')").format(
            value=', '.join(arguments[:-2]), bin=bins_size,
            start_or_end=field_name)
        return result

    def get_when_function(self, spec, params):
        """
        Map when() function call in Lemonade into when() call in Spark.
        """
        arguments = [self.parse(x, params) for x in spec['arguments']]
        # print >> sys.stderr, group(arguments[:-1], 2)
        code = ["when({}, {})".format(cond, value) for cond, value in
                group(arguments[:-1], 2)]
        if arguments[-1] is not None:
            code.append("otherwise({})".format(arguments[-1]))

        return '.'.join(code)

    def get_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        callee = spec['arguments'][0].get('callee', {})
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
        # Python as a UDF.
        custom_functions = {
            'group_datetime': self.get_window_function,
            'strip_accents': self.get_function_call,
            'strip_punctuation': self.get_function_call,
            'when': self.get_when_function,
            'window': self.get_window_function,
        }

        self.functions.update(spark_sql_functions)
        self.functions.update(custom_functions)
