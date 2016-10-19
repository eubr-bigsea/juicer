import json

class Expression:

    def __init__(self, json_code):
        self.code =  json.loads(json_code)
        self.right = ''
        self.left = ''
        self.expression = self.parse(self.code)


    def parse(self, tree):
        # Binary Expression parsing
        if tree['type'] == 'BinaryExpression':
            string = "(" + self.parse(tree['left']) + tree['operator'] + self.parse(tree['right']) + ")"
            return string
        # Literal parsing
        elif tree['type'] == 'Literal':
            return str(tree['value'])
        # Expression parsing
        elif tree['type'] == 'CallExpression':
            string = tree['callee']['name'] + "("
            for i in range(0, len(tree['arguments']) - 1):
                string += self.parse(tree['arguments'][i]) + ","
            string += self.parse(tree['arguments'][len(tree['arguments']) - 1]) + ")"
            return string
        # Identifier parsing
        elif tree['type'] == 'Identifier':
            return str(tree['name'])
        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            string = "(" + tree['operator'] + self.parse(tree['argument']) + ")"
            return string

    def print_expression(self):
        print self.expression


class Function:

    def __init__(self):
        self.code = ""

