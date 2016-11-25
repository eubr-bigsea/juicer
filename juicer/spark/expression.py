import json


class Expression:
    def __init__(self, json_code):
        self.code = json_code
        self.functions = {}
        self.build_functions_dict()

        self.parsed_expression = self.parse(json_code)


    def parse(self, tree):

        #print "\n\n",tree,"\n\n"
        # Binary Expression parsing
        if tree['type'] == 'BinaryExpression':
            string = "(" + self.parse(tree['left']) + tree[
                'operator'] + self.parse(tree['right']) + ")"
            return string

        # Literal parsing
        elif tree['type'] == 'Literal':
            return str("'" + str(tree['value']) + "'")

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            string = self.functions[tree['callee']['name']](tree)
            print string
            return string

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            return str("col('" + tree['name'] + "')")

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            string = "(" + tree['operator'] + self.parse(tree['argument']) + ")"
            return string


    # @staticmethod
    # def timestamp_to_datetime():
    #     target_columns = 'DTHR'
    #     new_name = 'DATETIME'
    #     format = '%Y-%m-%d %H:%M:%S'
    #     div = '1e6'
    #     code = """
    #         func = udf(lambda x: datetime.fromtimestamp(float(x)/{}).strftime({}))
    #     """.format(div, format)
    #     code += """
    #         {} = {}.withColumn({}, func(col({})))
    #     """.format(df1, df2, new_name, target_columns)
    #     return code
    #
    #
    # @staticmethod
    # def datetime_to_bins(spec):
    #     target_column = 'DATETIME'
    #     new_name = 'BINS_5_MIN'
    #     seconds = 300
    #     code = """
    #         {} = {}.withColumn({}, window({}, {}).start.cast('string'))
    #     """.format(df1, df2, new_name, target_column, seconds)
    #     return code


    def get_window_function(self, spec):
        """
        Window funciton is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        bins_size = arguments[-2][1:-1] + ' seconds'

        # COLOCAR A PALAVRA SECONDS DEPOIS DO PARAMETRO SEGUNDOS
        result = """window({}, '{}').{}.cast('timestamp')""".format(
            ', '.join(arguments[:-2]), bins_size, field_name)
        return result


    def get_function_call(self, spec, name=None):
        arguments = ', '.join([self.parse(x) for x in spec['arguments']])
        result = '{}({})'.format(spec['callee']['name'], arguments)
        return result


    def build_functions_dict(self):
        functions = {
            'regexp_replace': self.get_function_call,
            'to_date': self.get_function_call,
            'window': self.get_window_function,
            'group_datetime': self.get_window_function
        }
        self.functions.update(functions)


    # @staticmethod
    # def build_functions_dict():
    #     functions = {
    #         'regexp_replace': Expression.get_function_call,
    #         'to_date': self.get_function_call,
    #         'window': self.get_window_function,
    #         'timestamp_to_datetime': self.timestamp_to_datetime,
    #         'datetime_to_bins':Expression.datetime_to_bins
    #     }
    #     return functions



