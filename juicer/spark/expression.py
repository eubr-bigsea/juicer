import json


class Expression:
    def __init__(self, json_code):
        self.code = json_code
        self.functions = {}
        self.build_functions_dict()

        self.parsed_expression = self.parse(json_code)

    def parse(self, tree):

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
            return string

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            return str("col('" + tree['name'] + "')")

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            string = "(" + tree['operator'] + self.parse(tree['argument']) + ")"
            return string

    def get_window_function(self, spec):
        """
        Window funciton is slightly different from the Spark counterpart: the
        last parameter indicates if it is using the start or end field in
        window object. See Spark documentation about window. And a cast to
        timestamp is needed.
        """
        arguments = [self.parse(x) for x in spec['arguments']]

        field_name = 'start' if arguments[-1] != 'end' else 'end'
        result = """{}({}).{}.cast(timestamp')""".format(
            spec['callee']['name'],
            ', '.join(arguments[:-1]),
            field_name)
        return result

    def get_function_call(self, spec, name=None):

        arguments = ', '.join([self.parse(x) for x in spec['arguments']])
        result = '{}({})'.format(spec['callee']['name'], arguments)
        return result

    def build_functions_dict(self):
        functions = {
            'regexp_replace': self.get_function_call,
            'to_date': self.get_function_call,
            'window': self.get_window_function
        }
        self.functions.update(functions)
