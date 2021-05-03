#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from six import text_type


class ExpressionER:
    def __init__(self, json_code):
        self.code = json_code
        self.functions = {}
        self.imports_functions = {}
        self.translate_functions = {}
        self.build_functions_dict()

        self.parsed_expression = self.parse(json_code)

    def parse(self, tree):
        print(tree)
        # Expression parsing
        if tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            result = self.functions[tree['callee']['name']](tree)
        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def get_compare_function_call(self, spec):
        """
        Wrap column name with row() function call, if such call is not present.
        Convert the function to np.function.

        Example: sin(value) will be converted to np.sin(value)

        """

        # {'alias': '', 'expression': "exact('date_of_birth')", 'error': None, 'tree': {'type': 'CallExpression', 'arguments': [{'type': 'Literal', 'value': 'date_of_birth', 'raw': "'date_of_birth'"}], 'callee': {'
        #                                                                               type': 'Identifier', 'name': 'exact'}}}
        #                                                                               {'type': 'CallExpression', 'arguments': [{'type': 'Literal', 'value': 'date_of_birth', 'raw': "'date_of_birth'"}], 'callee': {'type': 'Identifier', 'name': 'exact'}}
        #                                                                                   {'type': 'Literal', 'value': 'date_of_birth', 'raw': "'date_of_birth'"}
        #
        #
        #                                                                               function = spec['callee']['name']

        function = spec['callee']['name']
        # Evaluates if column name is wrapped in a col() function call
        arguments = spec['arguments'][0]['raw']
        # function_name = spec['callee']['name']
        result = " compare.{}({})".format(function, arguments)
        return result

    def build_functions_dict(self):

        compare_functions = ['exact', 'string', 'numeric', 'date', 'geo']

        compare_functions = {k: self.get_compare_function_call
                                 for k in compare_functions}

        self.functions.update(compare_functions)