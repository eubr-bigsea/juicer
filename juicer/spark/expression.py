import sys

from juicer.util import group


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.build_functions_dict()
        self.parsed_expression = self.parse(json_code, params)

    def parse(self, tree, params):
        # print "\n\n",tree,"\n\n"
        # Binary Expression parsing
        if tree['type'] == 'BinaryExpression':
            return "{} {} {}".format(
                self.parse(tree['left'], params), tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            return tree['value']
            # return str("'" + str(tree['value']) + "'")

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            return self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                return "{}.{}".format(params['input'], tree['name'])
            else:
                return "col('{}')".format(tree['name'])

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            return "({} {})".format(tree['operator'],
                                    self.parse(tree['argument']))
        elif tree['type'] == 'LogicalExpression':
            operators = {
                "&&": "&",
                "||": "|",
                "!": "~"
            }
            operator = operators[tree['operator']]
            return "{} {} {}".format(self.parse(tree['left']),
                                     operator,
                                     self.parse(tree['right']))
        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

    def get_window_function(self, spec, params):
        """
        Window funciton is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        bins_size = '{} seconds'.format(arguments[-2])

        # COLOCAR A PALAVRA SECONDS DEPOIS DO PARAMETRO SEGUNDOS
        result = ("""window(from_unixtime(col({})/1e6), '{}')
                    .{}.cast('timestamp')""").format(
            ', '.join(arguments[:-2]), bins_size, field_name)
        return result

    def get_when_function(self, spec, params):
        """
        Map when() function call in Lemonade into when() call in Spark.
        """
        arguments = [self.parse(x) for x in spec['arguments']]
        # print >> sys.stderr, group(arguments[:-1], 2)
        code = ["when({}, {})".format(cond, value) for cond, value in
                group(arguments[:-1], 2)]
        if arguments[-1] is not None:
            code.append("otherwise({})".format(arguments[-1]))

        return '.'.join(code)

    def get_function_call(self, spec, params):
        """ Deprecated: use get_function_call_wrap_col instead """
        arguments = ', '.join([self.parse(x) for x in spec['arguments']])
        result = '{}({})'.format(spec['callee']['name'], arguments)
        return result

    def get_function_call_wrap_col(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        if 'input' not in params:
            if any([all([callee.get('type') == 'Identifier',
                         callee.get('name') == 'col']),
                    callee.get('type') == 'Identifier']):
                result = '{}({})'.format(spec['callee']['name'], arguments)
            else:
                result = '{}(col({}))'.format(spec['callee']['name'], arguments)
        else:
            result = '{}({}.{})'.format(spec['callee']['name'], params['input'],
                                        arguments)
        return result

    def build_functions_dict(self):
        spark_sql_functions = {
            'col': self.get_function_call_wrap_col,
            'concat': self.get_function_call_wrap_col,
            'concat_ws': self.get_function_call,
            'date_format': self.get_function_call_wrap_col,
            'length': self.get_function_call_wrap_col,
            'lit': self.get_function_call,
            'lower': self.get_function_call_wrap_col,
            'regexp_extract': self.get_function_call,
            'regexp_replace': self.get_function_call,
            'to_date': self.get_function_call_wrap_col,
            'when': self.get_when_function,
            'window': self.get_window_function,

        }
        # Functions that does not exist on Spark, but can be implemented in
        # Python as a UDF.
        custom_functions = {
            'group_datetime': self.get_window_function,
            'strip_accents': self.get_function_call_wrap_col,
            'strip_punctuation': self.get_function_call_wrap_col,
        }

        self.functions.update(spark_sql_functions)
        self.functions.update(custom_functions)
