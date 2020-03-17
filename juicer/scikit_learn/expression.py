#!/usr/bin/python
# -*- coding: utf-8 -*-

from six import text_type


class Expression:
    def __init__(self, json_code, params):
        self.code = json_code
        self.functions = {}
        self.imports_functions = {}
        self.translate_functions = {}
        self.build_functions_dict()

        self.imports = ""
        self.parsed_expression = "lambda row: " + self.parse(json_code, params)

    def parse(self, tree, params):

        if tree['type'] == 'BinaryExpression':
            result = "{} {} {}".format(
                self.parse(tree['left'], params),
                tree['operator'],
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            result = "'{}'".format(v) if isinstance(
                v, (str, text_type)) else str(v)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            result = self.functions[tree['callee']['name']](tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            if 'input' in params:
                result = "{}['{}']".format('row', tree[
                    'name'])  # params['input'], tree['name'])
            else:
                result = "functions.row['{}']".format(tree['name'])

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = '~'
            result = "({} {})".format(tree['operator'],
                                      self.parse(tree['argument'], params))

        elif tree['type'] == 'LogicalExpression':
            operators = {"&&": "&", "||": "|", "!": "~"}
            operator = operators[tree['operator']]
            result = "({}) {} ({})".format(self.parse(tree['left'], params),
                                           operator,
                                           self.parse(tree['right'], params))

        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [tree['test'], tree['consequent'],
                                  tree['alternate']]}
            result = self.get_when_function(spec, params)

        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def get_numpy_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.
        Convert the function to np.function
        """
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        # function_name = spec['callee']['name']
        result = " np.{}({})".format(function, arguments)
        return result

    def get_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.
        Custom functions or like "len()" or "str()"
        """
        callee = spec['arguments'][0].get('callee', {})
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        function_name = spec['callee']['name']
        result = "{}({})".format(function_name, arguments)
        return result

    def get_date_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.
        """
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        # origin = args[0]

        arguments = ', '.join(args)

        result = "{}({})".format(function, arguments)
        return result

    def get_built_function_call(self, spec, params):
        """
        Wrap column name with row() function call, if such call is not present.
        Functions like row().split()
        """
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        function = self.translate_functions[
            function] if function in self.translate_functions else function
        # Evaluates if column name is wrapped in a row() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        col = args[0]
        arguments = ', '.join(args[1:])

        result = "{}.{}({})".format(col, function, arguments)
        return result

    def get_date_instance_function_call(self, spec, params):
        """
        Wrap column name with col() function call, if such call is not present.
        """
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        args = [self.parse(x, params) for x in spec['arguments']]
        origin = args[0]

        arguments = ', '.join(args[1:])

        result = "{}.{}({})".format(origin, function, arguments)
        return result

    def get_window_function(self, spec, params):
        """
            Window function: group a datetime in bins
        """
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        args = [self.parse(x, params) for x in spec['arguments']]
        bins_size = args[1]
        var_date = args[0]

        result = "group_datetime(row[{date}], {bins_size})" \
            .format(date=var_date, bins_size=bins_size)

        return result

    def get_strip_accents_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        result = " ''.join(c for c in unicodedata.normalize('NFD', unicode({})) if unicodedata.category(c) != 'Mn')".format(
            arguments)

        return result

    def get_strip_punctuation_function(self, spec, params):
        callee = spec['arguments'][0].get('callee', {})
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp
        # Evaluates if column name is wrapped in a col() function call
        arguments = ', '.join(
            [self.parse(x, params) for x in spec['arguments']])
        strip_punctuation = ".translate(None, string.punctuation)"
        result = '{}{}'.format(arguments, strip_punctuation)
        return result

    def build_functions_dict(self):

        numpy_functions = {
            # ----- Mathematical operations -------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html

            # Trigonometric functions:
            'sin': self.get_numpy_function_call,
            'cos': self.get_numpy_function_call,
            'tan': self.get_numpy_function_call,
            'arcsin': self.get_numpy_function_call,
            'arccos': self.get_numpy_function_call,
            'arctan': self.get_numpy_function_call,
            'hypot': self.get_numpy_function_call,
            'arctan2': self.get_numpy_function_call,
            'deg2rad': self.get_numpy_function_call,
            'rad2deg': self.get_numpy_function_call,

            # Hyperbolic functions:
            'sinh': self.get_numpy_function_call,
            'cosh': self.get_numpy_function_call,
            'tanh': self.get_numpy_function_call,
            'arccosh': self.get_numpy_function_call,
            'arcsinh': self.get_numpy_function_call,
            'arctanh': self.get_numpy_function_call,

            # Rounding:
            'around': self.get_numpy_function_call,
            'rint': self.get_numpy_function_call,
            'fix': self.get_numpy_function_call,
            'floor': self.get_numpy_function_call,
            'ceil': self.get_numpy_function_call,
            'trunc': self.get_numpy_function_call,

            # Exponents and logarithms:
            'exp': self.get_numpy_function_call,
            'expm1': self.get_numpy_function_call,
            'exp2': self.get_numpy_function_call,
            'log': self.get_numpy_function_call,
            'log10': self.get_numpy_function_call,
            'log2': self.get_numpy_function_call,
            'log1p': self.get_numpy_function_call,
            'logaddexp': self.get_numpy_function_call,
            'logaddexp2': self.get_numpy_function_call,

            # Arithmetic operations:
            'add': self.get_numpy_function_call,
            'reciprocal': self.get_numpy_function_call,
            'negative': self.get_numpy_function_call,
            'multiply': self.get_numpy_function_call,
            'divide': self.get_numpy_function_call,
            'power': self.get_numpy_function_call,
            'subtract': self.get_numpy_function_call,
            'true_divide': self.get_numpy_function_call,
            'floor_divide': self.get_numpy_function_call,
            'float_power': self.get_numpy_function_call,
            'fmod': self.get_numpy_function_call,
            'remainder': self.get_numpy_function_call,

            # Miscellaneous
            'clip': self.get_numpy_function_call,
            'sqrt': self.get_numpy_function_call,
            'cbrt': self.get_numpy_function_call,
            'square': self.get_numpy_function_call,
            'fabs': self.get_numpy_function_call,
            'sign': self.get_numpy_function_call,
            'nan_to_num': self.get_numpy_function_call,

            # --------- String operations ---------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

            # String operations
            'mod': self.get_numpy_function_call,
            'capitalize': self.get_numpy_function_call,
            'center': self.get_numpy_function_call,
            'decode': self.get_numpy_function_call,
            'encode': self.get_numpy_function_call,
            'join': self.get_numpy_function_call,
            'ljust': self.get_numpy_function_call,
            'lower': self.get_numpy_function_call,
            'lstrip': self.get_numpy_function_call,
            'partition': self.get_numpy_function_call,
            'replace': self.get_numpy_function_call,
            'rjust': self.get_numpy_function_call,
            'rpartition': self.get_numpy_function_call,
            'rsplit': self.get_numpy_function_call,
            'rstrip': self.get_numpy_function_call,
            'splitlines': self.get_numpy_function_call,
            'strip': self.get_numpy_function_call,
            'swapcase': self.get_numpy_function_call,
            'upper': self.get_numpy_function_call,
            'zfill': self.get_numpy_function_call,
            'split': self.get_numpy_function_call,

            # String information
            'count': self.get_numpy_function_call,
            'find': self.get_numpy_function_call,
            'isalpha': self.get_numpy_function_call,
            'isdecimal': self.get_numpy_function_call,
            'isdigit': self.get_numpy_function_call,
            'islower': self.get_numpy_function_call,
            'isnumeric': self.get_numpy_function_call,
            'isspace': self.get_numpy_function_call,
            'istitle': self.get_numpy_function_call,
            'isupper': self.get_numpy_function_call,

            # --------- Logic operations ----------#
            # See more at:
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.logic.html

            # Logical operations
            'logical_and': self.get_numpy_function_call,
            'logical_or': self.get_numpy_function_call,
            'logical_not': self.get_numpy_function_call,
            'logical_xor': self.get_numpy_function_call,

            # Comparison
            'array_equiv': self.get_numpy_function_call,
            'equal': self.get_numpy_function_call,
            'not_equal': self.get_numpy_function_call,
            'greater_equal': self.get_numpy_function_call,
            'less_equal': self.get_numpy_function_call,
            'greater': self.get_numpy_function_call,
            'less': self.get_numpy_function_call,
        }

        date_functions = {

            #  -------- Date and time operations
            # See more at: https://docs.python.org/2/library/datetime.html

            'timedelta': self.get_date_function_call,

            'str2time': self.get_date_function_call,

            'date': self.get_date_function_call,
            'today': self.get_date_function_call,
            'now': self.get_date_function_call,
            'utcnow': self.get_date_function_call,
            'fromtimestamp': self.get_date_function_call,
            'utcfromtimestamp': self.get_date_function_call,
            'fromordinal': self.get_date_function_call,
            'combine': self.get_date_function_call,
        }

        date_instance_functions = {
            #  -------- Date and time operations
            # See more at: https://docs.python.org/2/library/datetime.html

            'toordinal': self.get_date_instance_function_call,
            'weekday': self.get_date_instance_function_call,
            'isoweekday': self.get_date_instance_function_call,
            'isoformat': self.get_date_instance_function_call,
            'total_seconds': self.get_date_instance_function_call,
        }

        self.functions.update(numpy_functions)
        self.functions.update(date_functions)
        self.functions.update(date_instance_functions)

        translate_functions = {

            'timedelta': 'datetime.timedelta',

            'str2time': 'parser.parse',

            'date': 'datetime.date',
            'today': 'date.today',
            'now': 'datetime.now',
            'utcnow': 'datetime.utcnow',
            'fromtimestamp': 'date.fromtimestamp',
            'utcfromtimestamp': 'date.utcfromtimestamp',
            'fromordinal': 'date.fromordinal',
            'combine': 'date.combine',
            'length': 'len',

            # Numpy string operations
            'strip': 'char.strip',
            'split': 'char.split',
            'mod': 'char.mod',
            'capitalize': 'char.capitalize',
            'center': 'char.center',
            'decode': 'char.decode',
            'encode': 'char.encode',
            'join': 'char.join',
            'ljust': 'char.ljust',
            'lower': 'char.lower',
            'lstrip': 'char.lstrip',
            'partition': 'char.partition',
            'replace': 'char.replace',
            'rjust': 'char.rjust',
            'rpartition': 'char.rpartition',
            'rsplit': 'char.rsplit',
            'rstrip': 'char.rstrip',
            'splitlines': 'char.splitlines',
            'strip': 'char.strip',
            'swapcase': 'char.swapcase',
            'upper': 'char.upper',
            'zfill': 'char.zfill',

            # Numpy string information
            'count': 'char.count',
            'find': 'char.find',
            'isalpha': 'char.isalpha',
            'isdecimal': 'char.isdecimal',
            'isdigit': 'char.isdigit',
            'islower': 'char.islower',
            'isnumeric': 'char.isnumeric',
            'isspace': 'char.isspace',
            'istitle': 'char.istitle',
            'isupper': 'char.isupper'
        }
        self.translate_functions.update(translate_functions)

        self.imports_functions = {
            "str2time": "from dateutil import parser",
            "strip_accents": "import unicodedata",
            "strip_punctuation": "import string",

        }

        # Functions that does not exist in Python, but can be implemented as a
        # lambda function. For now, and due simplicity, we require that every
        # custom function is necessarily defined here.
        others_functions = {
            'group_datetime': self.get_window_function,
            'strip_accents': self.get_strip_accents_function,
            'strip_punctuation': self.get_strip_punctuation_function,
            'str': self.get_function_call,
            'length': self.get_function_call,
            'len': self.get_function_call
        }
        self.functions.update(others_functions)
