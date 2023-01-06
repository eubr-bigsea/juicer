#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass, field
from gettext import gettext
from typing import Dict, List, Tuple

import polars as pl
from six import text_type

import juicer.scikit_learn.expression as sk_expression
from juicer.scikit_learn.util import (levenshtein, soundex, strip_accents,
                                      to_json)
from juicer.util import group


@dataclass
class SupportedFunction:
    """Class for store information about supported functions ."""
    name: str
    parameters: Tuple[type]
    parse: callable
    imports: List[str] = field(default_factory=list)
    custom_functions: Dict[str, callable] = field(default_factory=dict)


JAVA_2_PYTHON_DATE_FORMAT = {
    'yyyy': '%Y', 'y': '%Y', 'yy': '%y',
    'YYYY': '%Y', 'Y': '%Y', 'YY': '%y',

    'MM': '%m', 'MMM': '%b', 'MMMM': '%B', 'M': '%-m',

    'd': '%-d',
    'dd': '%d',
    'D': '%j',
    'HH': '%H', 'h': '%I',
    'mm': '%M',
    'ss': '%S', 's': '%-S',
    'S': '%s',
    'EEE': '%a', 'EEEE': '%A',
    'a': '%p',
    'Z': '%Z',
    "'": ''
}


class Expression(sk_expression.Expression):
    # __slots__ = ['imports']
    def __init__(self, json_code, params):
        super().__init__(json_code, params, False)
        # self.parsed_expression = self.parse(json_code, params)
        self.custom_functions: Dict[str, callable] = {}
        self.imports: List[str] = []

    def parse(self, tree, params):

        if tree['type'] == 'BinaryExpression':
            # Parenthesis are needed in some pandas/np expressions
            left = self.parse(tree['left'], params)
            right = self.parse(tree['right'], params)
            result = f"({left} {tree['operator']} {right})"

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            if isinstance(v, (str, text_type)):
                result = f"'{v}'"
            elif v is None:
                result = None
            else:
                result = str(v)

        # Expression parsing
        elif tree['type'] == 'CallExpression':
            if tree['callee']['name'] not in self.functions:
                raise ValueError(_('Function {f}() does not exists.').format(
                    f=tree['callee']['name']))
            fn: SupportedFunction = self.functions[tree['callee']['name']]
            self.custom_functions.update(fn.custom_functions)
            self.imports.extend(fn.imports)
            result = fn.parse(tree, params)

        # Identifier parsing
        elif tree['type'] == 'Identifier':
            result = tree['name']

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = 'not'
            result = "({} {})".format(tree['operator'],
                                      self.parse(tree['argument'], params))

        elif tree['type'] == 'LogicalExpression':
            operators = {"&&": "AND", "||": "OR", "!": "NOT"}
            operator = operators[tree['operator']]
            part1 = self.parse(tree['left'], params)
            part2 = self.parse(tree['right'], params)
            result = f"{part1} {operator} {part2}"

        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [tree['test'], tree['consequent'],
                                  tree['alternate']]}
            result = self.get_when_function(spec, params)

        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def _arg(self, tree: dict, params: dict, inx: int):
        """Returns the i-th parsed argument of the expression 
        Args:
            tree (dict): Expression tree
            params (dict): Expression parameters
            inx (int): Argument index

        Returns:
            _type_: Parsed expression for argument
        """
        return self.parse(tree['arguments'][inx], params)

    def _raw(self, tree: dict, params: dict, inx: int):
        """Returns the i-th value, without parsing
        Args:
            tree (dict): Expression tree
            params (dict): Expression parameters
            inx (int): Argument index

        Returns:
            _type_: Parsed expression for argument
        """
        return tree['arguments'][inx]['value']

    # def _get_python_date_format(self, fmt):
    #     parts = re.split(r'(\W)', fmt.replace('$', ''))
    #     return ''.join([JAVA_2_PYTHON_DATE_FORMAT.get(x, x)
    #                     for x in parts])

    def _when_call(self, spec, params):
        arguments = [self.parse(x, params) for x in spec['arguments']]

        code = [f"when({cond}).then({value})" for cond, value in
                group(arguments[:-1], 2)]

        if arguments[-1] is not None and (len(arguments) % 2 == 1):
            code.append(f"otherwise({arguments[-1]})")

        return 'pl.' + ('.'.join(code))

    # def _find_call(self, spec, params):
    #     function = spec['callee']['name']
    #     args = [self.parse(x, params) for x in spec['arguments']]
    #     pos = {'instr': 0, 'locate': args[2]}[function]

    #     return f"{args[0]}.apply(lambda x: x.find('{args[1]}', {pos}) + 1)"

    # def _substring_index_call(self, spec, params):
    #     args = [self.parse(x, params) for x in spec['arguments']]
    #     if args[2] >= 0:
    #         return f"{args[0]}.apply(lambda x: x.split('{args[1]}')[:{args[2]}]"
    #     else:
    #         return f"{args[0]}.apply(lambda x: x.split('{args[1]}')[{args[2]}:]"

    def _to_timestamp_call(self, spec: dict, params: dict, as_unix: bool = False,
                           use_tz: bool = False, use_date=False):
        # No parsing
        if len(spec['arguments']) > 1:
            fmt = self._get_python_date_format(spec['arguments'][1]['value'])
        else:
            fmt = '%Y-%m-%d %H:%M:%S'
        value = self.parse(spec['arguments'][0], params)

        if (as_unix):
            return f"STRPTIME({value}, '{fmt}')"
        elif use_tz:
            return f"TIMEZONE('UTC', STRPTIME({value}, '{fmt}'))"
        elif use_date:
            return f"CAST(STRPTIME({value}, '{fmt}') AS DATE)"
        else:
            return f"STRPTIME({value}, '{fmt}')"

    # def _array_set_call(self, spec: dict, params: dict) -> str:
    #     columns = ["'" + x['name'] + "'" for x in spec['arguments']]
    #     # args = [self.parse(x, params) for x in spec['arguments']]
    #     f = spec['callee']['name']
    #     m = {
    #         'array_except': 'difference',
    #         'array_intersect': 'intersection',
    #         'array_union': 'union'
    #     }[f]

    #     return (f"pl.struct([{columns[0]}, {columns[1]}]).apply("
    #             f"lambda x: list(set(x[{columns[0]}]).{m}(set(x[{columns[1]}]))))")

    def _function_call(self, spec: dict, params: dict, alias: str = None,
                       *args: any) -> str:
        parsed = [self.parse(x, params)
                  for i, x in enumerate(spec['arguments'])]
        function = (alias or spec['callee']['name']).upper()

        arguments = ', '.join(parsed[1:] + list(args))
        if len(arguments):
            result = f"{function}({parsed[0]}, {arguments})"
        else:
            result = f"{function}({parsed[0]})"
        return result

    def _function_call_fmt(self, spec: dict, params: dict, fmt: str = None
                           ) -> str:
        # if spec['callee']['name'] == 'expm1':
        #    import pdb; pdb.set_trace()
        parsed = [self.parse(x, params, i == 0)
                  for i, x in enumerate(spec['arguments'])]
        return fmt.format(*parsed)

    def _initcap_call(self, spec: dict, params: dict) -> str:
        parsed = [self.parse(x, params, i == 0)
                  for i, x in enumerate(spec['arguments'])]

        result = (f"LIST_AGGR(LIST_TRANSFORM(SPLIT(LOWER({parsed[0]}), ' '), "
                  f"x -> CONCAT(UPPER(SUBSTRING(x, 1, 1)), SUBSTRING(x, 2))), "
                  f"'STRING_AGG', ' ')")
        return result

    def _when_call(self, spec, params):
        arguments = [self.parse(x, params) for x in spec['arguments']]

        code = [f"WHEN({cond}) THEN {value}" for cond, value in
                group(arguments[:-1], 2)]

        if arguments[-1] is not None and (len(arguments) % 2 == 1):
            code.append(f"ELSE {arguments[-1]}")

        return f"(CASE {' '.join(code)} END)"

    def build_functions_dict(self):
        SF = SupportedFunction
        #number_regex: str = r'([\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+)'
        f1 = [
            SF('abs', (any,), self._function_call),
            SF('acos', (any,), self._function_call),
            SF('asin', (any,), self._function_call),
            SF('atan', (any,), self._function_call),
            SF('atan2', (any, any), self._function_call),
            # bround
            SF('cbrt', (any, any), self._function_call),
            SF('ceil', (any,), self._function_call),
            SF('cos', (any,), self._function_call),
            SF('cosh', (any,), lambda s, p:
                self._function_call_fmt(s, p, '0.5*(EXP(0.3)+EXP(-0.3))')),
            # SF('crc32', (any, ), lambda s, p: self._series_apply_call(
            #     s, p,
            #     fn="lambda x: zlib.crc32(x.encode('utf-8'))"),
            #     ['import zlib']),
            SF('degrees', (any, ), self._function_call),
            SF('exp', (any,), self._function_call),
            SF('expm1', (any,), lambda s, p:
                self._function_call_fmt(s, p, 'EXP({0}) - 1')),
            SF('floor', (any,), self._function_call),
            SF('hypot', (any, str, ), lambda s, p: self._function_call_fmt(
                s, p,  'SQRT({0}**2 + {1}**2)')),

            SF('log', (float, any,),
                lambda s, p: self._function_call_fmt(s, p,
                                                     'LN({0})' if len(s['arguments']) == 1
                                                     else 'LN({0})/LN({1})')),
            SF('log2', (any,), self._function_call),
            SF('log10', (any,), self._function_call),
            SF('log1p', (any,), lambda s, p:
                self._function_call_fmt(s, p, 'LOG(1 + {0})')),
            SF('pow', (any, float), self._function_call),
            SF('radians', (any, ), self._function_call),
            SF('rand', (any, ), lambda s, p: self._function_call_fmt(
                s, p,
                'COALESCE(SETSEED({0}), RANDOM()' if len(s['arguments']) == 1
                else 'RANDOM()')),
            # SF('randn', (any, ), lambda s, p: self._series_new_column_call(
            #    s, p, fn='lambda x: np.random.randn()')),
            # SF('rint', (any,), lambda s, p:
            #     self._function_call(s, p, 'round')),
            SF('round', (any,), self._function_call),
            # FIXME Array support
            #SF('sequence', (any,), self._function_call),

            SF('sin', (any,), self._function_call),
            SF('sinh', (any,), lambda s, p:
                self._function_call_fmt(s, p, '0.5*(EXP({0})-EXP(-{0}))')),
            SF('sqrt', (any,), self._function_call),
            SF('tan', (any,), self._function_call),
            SF('tanh', (any,), lambda s, p:
                self._function_call_fmt(s, p,
                                        '(EXP({0})-EXP(-{0}))/(EXP({0})+EXP({0}))')),

            # Bitwise
            SF('bitwiseNot', (any, ),
                lambda s, p: self._function_call_fmt(s, p, '~{0}')),
            SF('shiftLeft', (any, int),
                lambda s, p: self._function_call_fmt(s, p, '{0}<<{1}')),
            SF('shiftRight', (any, int),
                lambda s, p: self._function_call_fmt(s, p, '{0}>>{1}')),
            # SF('shiftRightUnsigned', (any, int),
            #     lambda s, p: self._bitwise_call(s, p, fn='right_shift')),

            # Logical/Conditional
            SF('isnull', (any, ),
               lambda s, p: self._function_call_fmt(s, p, '({0} IS NULL)')),
            SF('isnan', (any, ), self._function_call),

            # Date and time
            SF('add_months', (any, int), lambda s, p: self._function_call_fmt(
                s, p, '{0} + INTERVAL ({1}) MONTH')),
            SF('current_date', (any, ), self._function_call),
            SF('current_timestamp', (any, ),
               lambda s, p: self._function_call(s, p, 'get_current_timestamp')),
            SF('date_add', (any, int), lambda s, p: self._function_call_fmt(
                s, p, '{0} + INTERVAL ({1}) DAY')),
            SF('date_format', (any, str),
                lambda s, p: self._function_call(s, p, 'dt.strftime')),
            SF('date_sub', (any, int), lambda s, p: self._function_call_fmt(
                s, p, '{0} - INTERVAL ({1}) DAY')),
            SF('date_trunc', (str, any), self._function_call),
            SF('datediff', (any, str), lambda s, p: self._function_call_fmt(
                s, p, 'DATEDIFF("DAY", {0}, {1})')),
            SF('dayofmonth', (any, ), self._function_call),
            SF('dayofweek', (any, ), self._function_call),
            SF('dayofyear', (any,), self._function_call),
            SF('from_unixtime', (any,), lambda s, p:
               self._function_call_fmt(s, p, 'epoch_ms({0})')),
            SF('from_utc_timestamp', (any, str), lambda s, p:
                self._function_call_fmt(
                    s, p,
                "{0} AT TIME ZONE {1} AT TIME ZONE 'GMT'")),
            SF('hour', (any, ), self._function_call),
            SF('last_day', (any, ), self._function_call),
            SF('minute', (any, ), self._function_call),
            SF('month', (any, ), self._function_call),
            SF('months_between', (any, str), lambda s, p:
                self._function_call(s, p,
                                    "DATEDIFF('MONTH', CAST({0} AS TIMESTAMP), "
                                    "CAST({1} AS TIMESTAMP))")),

            SF('now', (any, ), self._function_call),
            SF('next_day', (any, int), lambda s, p:
                self._function_call(s, p,  "{0} + INTERVAL ({1}) DAY")),
            SF('quarter', (any, ), self._function_call),
            SF('second', (any, ), self._function_call),
            SF('to_date', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, use_date=True)),

            SF('to_timestamp', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, False)),

            SF('to_utc_timestamp', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, use_tz=True)),
            SF('trunc', (str, any), lambda s, p:
                self._function_call(s, p, 'date_trunc')),
            SF('unix_timestamp', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, True)),
            SF('weekofyear', (any, ), self._function_call),
            SF('year', (any, ), self._function_call),

            # String
            SF('ascii', (any, any), self._function_call),
            SF('base64', (any, any), lambda s, p:
                self._function_call_fmt(s, p, 'BASE64(ENCODE({0}))')),
            # SF('basestring', (any, any), self._function_call),
            # SF('bin',(any, any), self._function_call),
            SF('concat', (any, any), self._function_call),
            SF('concat_ws', (str, any), self._function_call),
            # SF('conv', (any, str, str), lambda s, p: self._series_apply_call(
            # SF('decode', (any, str), lambda s, p: self._series_apply_call(
            #     s, p, fn="lambda x: x.encode('utf8').decode({0})")),
            # SF('encode', (any, str), lambda s, p: self._series_apply_call(
            #     s, p, fn="lambda x: x.encode({0})")),
            SF('format_number', (any, int),
               lambda s, p: self._function_call_fmt(
                s, p, "FORMAT('{{0:.2f}}', CAST({0} AS DOUBLE))")),
            #     s, p,
            #     fn="lambda x: np.base_repr(int(x, base={0}), base={2})")),
            # SF('hex', (any, any),
            #     lambda s, p: self._function_call(s, p, 'encode', 'hex')),
            SF('initcap', (any, int), self._initcap_call),
            SF('instr', (any, str), self._function_call),
            SF('length', (any, any), self._function_call),
            SF('levenshtein', (any, any), self._function_call),
            SF('locate', (any, str, int), lambda s, p:
                self._function_call_fmt(
                    s, p, "INSTR(SUBSTRING({0}, {2}), {1}) + ({2}-1)")),
            SF('lower', (any, any),  self._function_call),
            SF('lpad', (any, int, str), self._function_call),
            SF('ltrim', (any, ), self._function_call),
            SF('regexp_extract', (any, str, int), self._function_call),
            SF('regexp_replace', (any, str, str), lambda s, p:
                self._function_call_fmt(s, p,
                                        "regexp_replace({0}, {1}, {2}, 'g')")),
            SF('repeat', (any, int), self._function_call),

            # FIXME Support to array
            SF('reverse', (any, ), self._function_call),
            SF('rpad', (any, int, str), self._function_call),
            SF('rtrim', (any, ), self._function_call),

            # Also used when datatype is array/list
            SF('size', (any, any), lambda s, p:
                self._function_call(s, p, 'length')),

            # SF('soundex', (any, ), lambda s, p: self._series_apply_call(
            #     s, p, fn="lambda x: soundex(x)"),
            #     custom_functions={'soundex': soundex}),
            SF('split', (any,), self._function_call),
            SF('substring', (any, int, int), self._function_call),
            # SF('substring_index', (any, str, int), self._substring_index_call),
            SF('strip_accents', (any, ), self._function_call),

            # Next version
            # SF('translate', (any, str, str), self._function_call),
            SF('trim', (any,), self._function_call),
            SF('unbase64', (any, any),
               lambda s, p: self._function_call(s, p, 'from_base64')),
            # SF('unhex', (any, any),
            #     lambda s, p: self._function_call(s, p, 'decode', 'hex')),
            SF('upper', (any, any), self._function_call),

            # Utilities
            SF('coalesce', (any, any), self._function_call),
            # SF('create_map', (any,), self._function_call),

            SF('from_json', (any, ),
               lambda s, p: self._function_call(s, p, 'json')),
            SF('get_json_object', (any, str, ), lambda s, p:
                self._function_call(s, p, 'json_extract')),
            # SF('greatest', (any, List[any]), lambda s, p:
            #     self._struct_apply_call(
            #         s, p,
            #     'lambda x: np.max('
            #     '[v for v in x.values() if v is not None] or None)')),
            # SF('hash', (any, List[any]), lambda s, p:
            #    self._struct_apply_call(
            #     s, p,
            #     'lambda x: hash((v for v in x if))')),
            # SF('json_tuple', (any, any), self._function_call),
            # SF('least', (any, List[any]), lambda s, p:
            #     self._struct_apply_call(
            #         s, p,
            #     'lambda x: np.max('
            #     '[v for v in x.values() if v is not None] or None)')),
            SF('lit', (any,), lambda s, p:
                self._function_call_fmt(s, p, '{0}')),
            SF('md5', (any, ), self._function_call),
            # SF('monotonically_increasing_id', (any, ), self._function_call),
            SF('nanvl', (any, any), lambda s, p:
                self._function_call_fmt(
                    s, p,
                'CASE WHEN ISNAN({0}) THEN {1} ELSE {0} END')),
            SF('schema_of_json', (any,), lambda s, p:
                self._function_call(s, p, 'json_structure')),

            # SF('sha1', (any, ), lambda s, p: self._series_apply_call(
            #     s, p,
            #     fn="lambda x: hashlib.sha1(x.encode('utf8')).hexdigest()"),
            #     ['import hashlib']),
            # SF('sha2', (any, int), lambda s, p: self._series_apply_call(
            #     s, p,
            #     fn=f"lambda x: hashlib.sha{self.parse(s['arguments'][1], p)}"
            #     "(x.encode('utf8')).hexdigest()"),
            #     ['import hashlib']),

            SF('signum', (any,), lambda s, p:
                self._function_call(s, p, 'sign')),
            SF('to_json', (any, ), self._function_call),

            # Array/Vectors
            SF('array_contains', (any, any), self._function_call),
            SF('array_distinct', (any, ), self._function_call),
            # SF('array_except', (any, any), self._function_call),
            # SF('array_intersect', (any, any), self._function_call),
            # https://github.com/duckdb/duckdb/issues/1940
            # SF('array_join', (any, any),
            #     lambda s, p: self._function_call(s, p, 'arr.join')),

            SF('array_max', (any, ),
                lambda s, p: self._function_call(s, p, 'list_max')),
            SF('array_min', (any, ),
                lambda s, p: self._function_call(s, p, 'list_min')),
            SF('array_position', (any, any),
               lambda s, p: self._function_call(s, p, 'list_position')),
            SF('array_remove', (any, any),
               lambda s, p: self._function_call_fmt(
                s, p,
                'LIST_FILTER({0}, x -> x <> {1})')),
            # SF('array_repeat', (any, any),
            #    lambda s, p: self._function_call(s, p, 'arr.max')),
            SF('array_sort', (any, ), self._function_call),
            SF('sort_array', (any, ),
                lambda s, p: self._function_call(s, p, 'arr.sort')),
            SF('array_union', (any, any), lambda s, p:  self._function_call_fmt(
                s, p, 'LIST_DISTINCT(LIST_CONCAT({0}, {1}))')),
            # SF('arrays_overlap', (any, any),
            #    lambda s, p: self._function_call(s, p, 'arr.max')),
            # SF('arrays_zip', (any, any),
            #    lambda s, p: self._function_call(s, p, 'arr.max')),
            SF('element_at', (any, ),
                lambda s, p: self._function_call_fmt(s, p, '{0}[{1}]')),
            SF('explode', (any, ),
                lambda s, p: self._function_call(s, p, 'unnest')),
            # 'explode', 'explode_outer',
            # 'posexplode',
            # 'flatten',
            # 'map_concat', 'map_from_arrays', 'map_keys',
            # 'map_values',
            # SF('shuffle', (any, ), lambda s, p: self._series_apply_call(
            #     s, p,
            #     fn='lambda x: x.shuffle(random.randint(0, sys.maxsize))'),
            #     ['import random']),
            SF('slice', (any, int, int),
                lambda s, p: self._function_call(s, p, 'list_slice')),
            SF('sort_array', (any, bool), lambda s, p: self._function_call_fmt(
                s, p,
                '(CASE WHEN NOT {1} THEN LIST_SORT({0}}) '
                'ELSE LIST_REVERSE_SORT({0}) END)')),
            # Array size() is the same as the string function


            # Data Explorer
            SF('when', (any, List[any]), self._when_call),
            SF('isnotnull', (any, ), lambda s, p:
               self._function_call_fmt(s, p, '({0} IS NOT NULL)')),
            SF('cast', (any, any), lambda s, p:
               f"CAST({self._arg(s, p, 0)} AS {self._raw(s, p, 1)})"),
            SF('cast_array', (any, str), lambda s, p:
                self._function_call_fmt(
                    s, p, 'LIST_TRANSFORM({0}, X -> CAST(X AS {1}))')
               ),

            SF('extract_numbers', (any, ), lambda s, p: self._function_call_fmt(
                s, p,
                "LIST_TRANSFORM(LIST_FILTER("
                "STRING_SPLIT_REGEX({0}', '[^\D\.]'), X->X<>''), "
                "X->CAST(X AS DOUBLE))")
               ),

        ]

        # 'bround',
        # create_map',

        # 'shiftRightUnsigned',
        # 'json_tuple',
        # monotonically_increasing_id: NOT IMPLEMENTED
        # struct: NOT IMPLEMENTED,

        for f in f1:
            self.functions[f.name] = f
            # if f.imports:
            #     self.imports += '\n' + '\n'.join(f.imports)
            #     self.functions
            # import pdb; pdb.set_trace()
            # if f.custom_functions:
            #     self.custom_functions = f.custom_functions
