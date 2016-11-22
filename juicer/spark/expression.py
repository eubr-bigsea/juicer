import json


class Expression:
    def __init__(self, json_code):
        self.code = json_code
        self.functions = self.build_functions_dict()
        self.parsed_expression = self.parse(self.code)

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
            string = self.functions[tree['callee']['name']] + "("
            for i in range(0, len(tree['arguments']) - 1):
                string += self.parse(tree['arguments'][i]) + ","
            string += self.parse(
                tree['arguments'][len(tree['arguments']) - 1]) + ")"
            return string

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            return str("col('" + tree['name'] + "')")

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            string = "(" + tree['operator'] + self.parse(tree['argument']) + ")"
            return string

    def build_functions_dict(self):
        dict = {'REPLACE': 'regexp_replace'}
        return dict
