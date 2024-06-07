#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
    "'": '',
    '"': ''
}


class Expression(sk_expression.Expression):
    # __slots__ = ['imports']
    def __init__(self, json_code, params):
        sk_expression.Expression.__init__(self, json_code, params, False)
        # self.parsed_expression = self.parse(json_code, params)
        self.custom_functions: Dict[str, callable] = {}
        self.imports: List[str] = []

    def _arg(self, tree: dict, params: dict, inx: int, enclose_literal=False):
        """Returns the i-th parsed argument of the expression 
        Args:
            tree (dict): Expression tree
            params (dict): Expression parameters
            inx (int): Argument index
            enclose_literal (bool, optional): Enclose literal in quotes. 
                Defaults to False.

        Returns:
            _type_: Parsed expression for argument
        """
        return self.parse(tree['arguments'][inx], params, enclose_literal)

    def parse(self, tree, params, enclose_literal=False):

        if tree['type'] in ('BinaryExpression', 'LogicalExpression'):
            operators = {"&&": "&", "||": "|", "!": "~"}
            if tree['operator'] in operators:
                operator = operators[tree['operator']]
            else:
                operator = tree['operator']
            # Parenthesis are needed in some pandas/np expressions
            result = "({} {} {})".format(
                self.parse(tree['left'], params),
                operator,
                self.parse(tree['right'], params))

        # Literal parsing
        elif tree['type'] == 'Literal':
            v = tree['value']
            if isinstance(v, (str, text_type)):
                result = f"'{v}'"
            elif v is None:
                result = None
            else:
                result = str(v)
            if params.get('enclose_literal') or enclose_literal:
                result = f'pl.lit({result})'

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
            if 'input' in params:
                result = f"pl.col('{tree['name']}')"
            else:
                result = f"pl.col('{tree['name']}')"

        # Unary Expression parsing
        elif tree['type'] == 'UnaryExpression':
            if tree['operator'] == '!':
                tree['operator'] = 'not'
            result = "({} {})".format(tree['operator'],
                                      self.parse(tree['argument'], params))

        elif tree['type'] == 'ConditionalExpression':
            spec = {'arguments': [tree['test'], tree['consequent'],
                                  tree['alternate']]}
            result = self._when_call(spec, params)

        elif tree['type'] == 'ArrayExpression':
            result = f'pl.Series([{repr([e["value"] for e in tree["elements"]])}])'
        else:
            raise ValueError("Unknown type: {}".format(tree['type']))

        return result

    def _get_python_date_format(self, fmt):
        parts = re.split(r'(\W)', fmt.replace('$', ''))
        return ''.join([JAVA_2_PYTHON_DATE_FORMAT.get(x, x)
                        for x in parts])

    def _series_method_call(self, spec: dict, params: dict,
                            alias: str = None, *args: any) -> str:

        # import pdb; pdb.set_trace()
        function = spec['callee']['name']
        if function in self.imports_functions:
            imp = self.imports_functions[function] + "\n"
            if imp not in self.imports:
                self.imports += imp

        # if spec['callee']['name'] == 'sin':
        #    import pdb;pdb.set_trace()

        parsed = [self.parse(x, params, i == 0)
                  for i, x in enumerate(spec['arguments'])]
        function = alias or function

        map_names = {'trim': 'strip', 'rtrim': 'rstrip', 'ltrim': 'lstrip'}
        f = map_names.get(function, function)

        arguments = ', '.join(parsed[1:] + list(args))
        result = f"{parsed[0]}.{f}({arguments})"
        return result

    def _series_apply_call(self, spec: dict, params: dict,
                           fn: str = None) -> str:

        parsed = [self.parse(x, params, i == 0)
                  for i, x in enumerate(spec['arguments'])]

        if '{' in fn:
            fn = fn.format(*parsed)

        result = f"{parsed[0]}.apply({fn})"
        return result

    def _struct_apply_call(self, spec: dict, params: dict,
                           fn: str = None) -> str:

        parsed = [self.parse(x, params) for x in spec['arguments']]
        arguments = ', '.join(parsed)
        result = f"pl.struct([{arguments}]).apply({fn})"
        return result

    def _rand_call(self, spec: dict, params: dict, alias: str) -> str:

        # parsed = [self.parse(x, params) for x in spec['arguments']]
        result = f"pl.Series(np.random.{alias}({params['input']}.shape[0]))"
        return result

    # def _bitwise_call(self, spec: dict, params: dict,
    #                   fn: str = None) -> str:

    #     parsed = [self.parse(x, params) for x in spec['arguments']]
    #     result = f"{parsed[0]}.apply(lambda x: np.{fn}(x, {parsed[1]}))"
    #     return result

    def _pl_method_call(self, spec: dict, params: dict,
                        alias: str = None, *args: any) -> str:

        function = alias or spec['callee']['name']
        params['enclose_literal'] = function != 'lit'
        parsed = [self.parse(x, params) for x in spec['arguments']]
        del params['enclose_literal']

        if function == 'lit':
            arguments = ', '.join(parsed + list(args))
            result = f"pl.{function}({arguments})"
        else:
            arguments = ', '.join(parsed + list(args))
            result = f"pl.{function}([{arguments}])"
        return result

    def _add_month_call(self, spec: dict, params: dict) -> str:
        args = [self.parse(x, params) for x in spec['arguments']]
        return f"{args[0]}.dt.offset_by('{args[1]}mo')"

    def _last_day(self, spec: dict, params: dict) -> str:
        args = [self.parse(x, params) for x in spec['arguments']]
        return (f"{args[0]}.dt.offset_by('1mo').dt.round('1mo')"
                f" - pl.duration(days=1)")

    def _log_call(self, spec: dict, params: dict) -> str:
        args = [self.parse(x, params) for x in spec['arguments']]
        return f"{args[1]}.log({args[0]})"

    def _date_trunc_call(self, spec: dict, params: dict,
                         reversed_params: bool = False) -> str:
        args = [self.parse(x, params) for x in spec['arguments']]

        formats = {'year': '1y', 'yyyy': '1y', 'yy': '1y',
                   'month': '1mo', 'mon': '1mo', 'mm': '1mo',
                   'day': '1d', 'dd': '1d',
                   'hour': '1h', 'minute': '1m', 'second': '1s', 'week': '1w',
                   'quarter': '3mo'}
        if reversed_params:
            return f"{args[1]}.dt.truncate('{formats.get(args[0][1:-1])}')"
        else:
            return f"{args[0]}.dt.truncate('{formats.get(args[1][1:-1])}')"

    def _when_call(self, spec, params):
        arguments = [self.parse(x, params) for x in spec['arguments']]

        code = [f"when({cond}).then({value})" for cond, value in
                group(arguments[:-1], 2)]

        if arguments[-1] is not None and (len(arguments) % 2 == 1):
            code.append(f"otherwise({arguments[-1]})")

        return 'pl.' + ('.'.join(code))

    def _signum_call(self, spec, params):
        arguments = [self.parse(x, params) for x in spec['arguments']]
        return (f"pl.when({arguments[0]} > 0).then(1)"
                f".when({arguments[0]} < 0).then(-1).otherwise(0)")

    def _nanvl_call(self, spec, params):
        args = [self.parse(x, params) for x in spec['arguments']]
        return (f"pl.when({args[0]}.is_not_nan()).then({args[0]})"
                f".otherwise({args[1]})")

    def _datediff_call(self, spec: dict, params: dict, unit: str = 'days'):
        args = [self.parse(x, params) for x in spec['arguments']]
        return (f"({args[0] - args[1]}).dt.{unit}()")

    def _find_call(self, spec, params):
        function = spec['callee']['name']
        args = [self.parse(x, params) for x in spec['arguments']]
        pos = {'instr': 0, 'locate': args[2]}[function]

        return f"{args[0]}.apply(lambda x: x.find('{args[1]}', {pos}) + 1)"

    def _substring_index_call(self, spec, params):
        args = [self.parse(x, params) for x in spec['arguments']]
        if args[2] >= 0:
            return f"{args[0]}.apply(lambda x: x.split('{args[1]}')[:{args[2]}]"
        else:
            return f"{args[0]}.apply(lambda x: x.split('{args[1]}')[{args[2]}:]"

    def _to_timestamp_call(self, spec: dict, params: dict, as_unix: bool = False,
                           use_tz: bool = False, use_date=False):
        # No parsing
        fmt = self._get_python_date_format(spec['arguments'][1]['value'])
        value = self.parse(spec['arguments'][0], params)
        # Not working in Polars when day has a single digit (no padding)
        # )str.strptime(pl.Datetime, fmt='{fmt}', strict=False)"

        if (as_unix):
            return f"{value}.cast(pl.Int64)"
            # return (
            #     f"{value}.apply(lambda x: "
            #     f"int(datetime.datetime.strptime(x, '{fmt}').timestamp() * 1e3))"
            # )
        elif use_tz:
            return (f"{value}.apply(lambda x: "
                    f"datetime.datetime.strptime(x, '{fmt}')"
                    f".replace(tzinfo=datetime.timezone.utc))")
        elif use_date:
            return (f"{value}.apply(lambda x: "
                    f"datetime.datetime.strptime(x, '{fmt}'))")
        else:
            return (f"{value}.apply(lambda x: "
                    f"datetime.datetime.strptime(x, '{fmt}'))")

    def _array_set_call(self, spec: dict, params: dict) -> str:
        columns = ["'" + x['name'] + "'" for x in spec['arguments']]
        # args = [self.parse(x, params) for x in spec['arguments']]
        f = spec['callee']['name']
        m = {
            'array_except': 'difference',
            'array_intersect': 'intersection',
            'array_union': 'union'
        }[f]

        return (f"pl.struct([{columns[0]}, {columns[1]}]).apply("
                f"lambda x: list(set(x[{columns[0]}]).{m}(set(x[{columns[1]}]))))")

    def _split_call(self, spec: dict, params: dict) -> str:
        # Split does not support regex
        # https://github.com/pola-rs/polars/issues/4819
        args = [self.parse(x, params) for x in spec['arguments']]
        return f"{args[0]}.str.replace_all(r{args[1]}, chr(2)).str.split(chr(2))"

    def _concat_ws_call(self, spec: dict, params: dict) -> str:
        # First parameter is the deliter. In polars, it is the last
        args = spec['arguments'][1:] + spec['arguments'][0:1]
        new_spec = dict(spec)
        new_spec['arguments'] = args
        return self._pl_method_call(new_spec, params, 'concat_str')

    def _assert_int(self, description: str, value: int):
        if not isinstance(value, int):
            raise ValueError(_('{} is not integer').format(description, value))

    def _call_fmt(self, spec: dict, params: dict, fmt: str = None,
                  use_raw: set = None) -> str:
        if use_raw is None:
            use_raw = set()

        parsed = [self.parse(x, params, i == 0) if i not in use_raw
                  else x['raw']
                  for i, x in enumerate(spec['arguments'])]
        return fmt.format(*parsed)

    def build_functions_dict(self):
        SF = SupportedFunction
        number_regex: str = r'([\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+)'
        f1 = [

            # Bitwise
            SF('bitwiseNot', (any, ), lambda s, p: self._call_fmt(
                s, p, fmt='np.invert(pl.col({0}))'), ['import numpy as np']),
            SF('shiftLeft', (any, int), lambda s, p: self._call_fmt(
                s, p, fmt='np.left_shift({0}, {1})'), ['import numpy as np']),
            SF('shiftRight', (any, int), lambda s, p: self._call_fmt(
                s, p, fmt='np.right_shift({0}, {1})'), ['import numpy as np']),
            # SF('shiftRightUnsigned', (any, int), lambda s, p: self._call_fmt(
            #     s, p, fmt='np.right_shift({0}, {1})'), ['import numpy as np']),

            # Logic/conditional
            SF('isnull', (any, ),
               lambda s, p: self._series_method_call(s, p, 'is_null')),
            SF('isnan', (any, ),
               lambda s, p: self._series_method_call(s, p, 'is_nan')),
            SF('when', (any, List[any]), self._when_call),


            SF('abs', (any,), self._series_method_call),
            SF('acos', (any,),
                lambda s, p: self._series_method_call(s, p, 'arccos')),
            SF('asin', (any,),
                lambda s, p: self._series_method_call(s, p, 'arcsin')),
            SF('atan', (any,),
                lambda s, p: self._series_method_call(s, p, 'arctan')),
            # See atan2 definition
            SF('atan2', (any, float, float), lambda s, p:self._call_fmt(
                s, p,
                '''(
                    pl.when({1} > 0).then(({0} / {1}).arctan())
                    pl.when(({1} < 0) & ({0} >=0)).then(
                        ({0} / {1}).arctan() + math.pi)
                    pl.when(({1} < 0) & ({0} < 0)).then(
                        ({0} / {1}).arctan() - math.pi)
                    pl.when(({1} == 0) & ({0} > 0)).then(math.pi / 2)
                    pl.when(({1} == 0) & ({0} < 0)).then(- math.pi / 2)
                        .otherwise(0.0)
                )'''), ['import math']),
            SF('bround', (any, any), lambda s, p: self._series_apply_call(
                s, p,
                '''(lambda n: round(n) + 1 if math.isclose(math.modf(n)[0],0.5) 
                    and int(number) % 2 != 0 else round(number))'''),
               ['import math']),
            SF('cbrt', (any, ), lambda s, p:
                self._series_method_call(s, p, 'pow', '1.0/3')),
            # Alternative (using pow is faster)
            # SF('cbrt', (any, ), lambda s, p:
            #     self._call_fmt(s, p, np.cbrt({0}))),
            SF('ceil', (any,), self._series_method_call),
            SF('cos', (any,), self._series_method_call),
            SF('cosh', (any,), self._series_method_call),
            SF('crc32', (any, ), lambda s, p: self._series_apply_call(
                s, p,
                fn="lambda x: zlib.crc32(x.encode('utf-8'))"),
                ['import zlib']),
            SF('degrees', (any, ), lambda s, p: self._call_fmt(
                s, p, '({0} * 180.0) / math.pi'), imports=['import math']),
            SF('exp', (any,), self._series_method_call),
            SF('expm1', (any,), lambda s, p: self._call_fmt(s, p,
                                                            '{0}.exp() - 1')),
            SF('floor', (any,), self._series_method_call),
            SF('hypot', (any, str, ), lambda s, p: self._call_fmt(
                s, p, '({0}**2 + {1}**2).sqrt()')),
            SF('log', (float, any,), self._log_call),
            SF('log10', (any,), self._series_method_call),
            SF('log1p', (any,), self._series_method_call),
            # lambda s, p: self._call_fmt(
            # s, p, '({0} + 1).log()')),
            SF('log2', (any,), lambda s, p:
                self._series_method_call(s, p, 'log', '2')),
            SF('pow', (any, float), self._series_method_call),
            SF('radians', (any, ), lambda s, p: self._call_fmt(
                s, p, '({0} * math.pi) / 180.0'), imports=['import math']),

            SF('rand', (any, ), lambda s, p: self._rand_call(
                s, p, alias='rand')),
            SF('randn', (any, ), lambda s, p: self._rand_call(
                s, p, alias='randn')),
            SF('rint', (any,), lambda s, p:
                self._series_method_call(s, p, 'round')),
            SF('round', (any, int), self._series_method_call),
            # Sequence
            SF('sin', (any,), self._series_method_call),
            SF('sinh', (any,), self._series_method_call),
            SF('sqrt', (any,), self._series_method_call),
            SF('tan', (any,), self._series_method_call),
            SF('tanh', (any,), self._series_method_call),

            # Date and time
            SF('add_months', (any, int), self._add_month_call),
            SF('current_date', (any, ),
               lambda s, p: self._pl_method_call(s, p, 'lit',
                                                 'datetime.date.today()')),
            SF('current_timestamp', (any, ),
               lambda s, p: self._pl_method_call(s, p, 'lit',
                                                 'datetime.datetime.now()')),
            SF('date_add', (any, int),
                lambda s, p: self._series_method_call(s, p, 'dt.offset_by')),
            SF('date_format', (any, str),
                lambda s, p: self._series_method_call(s, p, 'dt.strftime')),
            SF('date_sub', (any, int),
                lambda s, p: self._series_method_call(s, p, 'dt.offset_by')),

            SF('date_trunc', (any, str), self._date_trunc_call),
            SF('datediff', (any, any), self._datediff_call),
            SF('dayofmonth', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.day')),
            SF('dayofweek', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.weekday')),
            SF('dayofyear', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.ordinal_day')),
            SF('from_unixtime', (any, int), lambda s, p:
               self._call_fmt(s, p,
                              '{0}.cast(pl.Datetime).dt.with_time_unit("ms")')),
            SF('from_utc_timestamp', (any, str), lambda s, p: self._call_fmt(
                s, p, '{0}.dt.tz_localize("UTC").dt.cast_time_zone("{1}")')),
            SF('hour', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.hour')),
            SF('last_day', (any, ), self._series_method_call),
            SF('minute', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.minute')),
            SF('month', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.month')),
            SF('months_between', (any, str), lambda s, p: self._call_fmt(
                s, p, '{0} - {1}).dt.days() / 30')),
            SF('next_day', (any, int),
               lambda s, p: self._series_method_call(s, p, 'str.offset_by')),
            SF('now', (any, ),
               lambda s, p: self._pl_method_call(s, p, 'lit',
                                                 'datetime.datetime.now()')),
            # SF('months_between', (any, any), lambda s, p:
            #     self._datediff_call(s, p, 'months')),
            SF('quarter', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.quarter')),
            SF('second', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.second')),
            SF('to_date', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, use_date=True)),
            SF('to_timestamp', (any, str), lambda s, p:
                self._to_timestamp_call(s, p, False)),
            SF('to_utc_timestamp', (any, str), lambda s, p: self._call_fmt(
                s, p, "'{0}'.dt.tz_localize('{1}').dt.cast_time_zone('UTC')")),
            # Reverso e com parâmetros diferentes mapeáveis
            SF('trunc', (str, any), self._date_trunc_call),
            SF('unix_timestamp', (any, str), lambda s, p: self._call_fmt(
                s, p, '{0}.cast(pl.Int64)')),
            SF('weekofyear', (any, any),
                lambda s, p: self._series_method_call(s, p, 'dt.week')),
            SF('year', (any, any),
               lambda s, p: self._series_method_call(s, p, 'dt.second')),

            # Text
            SF('ascii', (any, any), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: ord(x)")),
            SF('base64', (any, any),
                lambda s, p: self._series_method_call(s, p, 'encode', 'base64')),
            SF('bin', (any, any), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: bin(x)")),
            SF('concat', (any, any),
                lambda s, p: self._pl_method_call(s, p, 'concat_str')),
            SF('concat_ws', (any, any), self._concat_ws_call),
            SF('conv', (any, str, str), lambda s, p: self._series_apply_call(
                s, p,
                fn="lambda x: np.base_repr(int(x, base={1}), base={2})")),
            SF('decode', (any, str), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: x.encode('utf8').decode({1})")),
            SF('encode', (any, str), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: x.encode({1})")),
            SF('format_number', (any, int),
               lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: '{{:,.5f}}'.format(x)")),
            SF('hex', (any, any),
                lambda s, p: self._series_method_call(s, p, 'encode', 'hex')),

            SF('initcap', (any, int),
                lambda s, p: self._series_method_call(s, p, 'str.to_titlecase')),
               # lambda s, p: self._series_apply_call(
               # s, p, fn="str.title")),
            SF('instr', (any, str), self._find_call),
            SF('length', (any, any),
                lambda s, p: self._series_method_call(s, p, 'str.lengths')),
            SF('levenshtein', (any, ),
               lambda s, p: self._series_apply_call(
                s, p, fn="levenshtein"),
                custom_functions=levenshtein),

            SF('locate', (any, str, int), self._find_call),
            SF('lower', (any, any),
                lambda s, p:
                self._series_method_call(s, p, 'str.to_lowercase')),
            SF('lpad', (any, int, str),
                lambda s, p: self._series_method_call(s, p, 'str.ljust')),
            SF('ltrim', (any, ),
                lambda s, p: self._series_method_call(s, p, 'str.lstrip')),
            SF('regexp_extract', (any, str, int), lambda s, p: self._call_fmt(
                s, p, '{0}.str.extract_all(r{1})')),
            SF('regexp_replace', (any, str, str), lambda s, p:
                self._series_method_call(s, p, 'str.replace_all')),
            SF('repeat', (any, int), lambda s, p: self._series_apply_call(
                s, p, fn=f"lambda x: x * {self._arg(s, p, 1)}")
               ),
            SF('reverse', (any, ), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: x[::-1]")
               ),
            SF('rpad', (any, int, str),
                lambda s, p: self._series_method_call(s, p, 'str.rjust')),
            SF('rtrim', (any, ),
                lambda s, p: self._series_method_call(s, p, 'str.rstrip')),
            SF('soundex', (any, ), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: soundex(x)"),
                custom_functions={'soundex': soundex}),
            SF('strip_accents', (any, ), lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: strip_accents(x)"),
                imports=['import unicodedata'],
                custom_functions={'strip_accents': strip_accents}),
            # Separator is the first argument
            SF('split', (any,),  # https://github.com/pola-rs/polars/issues/4819
                self._split_call),
            SF('substring', (any, int, int),
                lambda s, p: self._series_method_call(s, p, 'cast(pl.Utf8).str.slice')),
            SF('substring_index', (any, str, int), self._substring_index_call),
            SF('translate', (any, str, str),
               lambda s, p: self._series_apply_call(
                s, p, fn="lambda x: x.translate(x.maketrans({1}, {2}))")),
            SF('trim', (any,),
                lambda s, p: self._series_method_call(s, p, 'str.strip')),
            SF('unbase64', (any, any),
               lambda s, p: self._series_method_call(s, p, 'decode', 'base64')),
            SF('unhex', (any, any),
                lambda s, p: self._series_method_call(s, p, 'decode', 'hex')),
            SF('upper', (any, any),
                lambda s, p: self._series_method_call(s, p,
                                                      'str.to_uppercase')),
            # SF('basestring', (any, any), self._series_method_call),

            # Util
            SF('coalesce', (any, any), self._pl_method_call),
            SF('from_json', (any, ),
               lambda s, p: self._series_apply_call(
                s, p, fn="json.loads"),
                imports=['import json']),
            SF('get_json_object', (any, str, ), lambda s, p:
                self._series_method_call(s, p, 'str.json_path_match')),
            # SF('greatest', (any, List[any]), lambda s, p:
            #     self._struct_apply_call(
            #         s, p,
            #     'lambda x: np.max('
            #     '[v for v in x.values() if v is not None] or None)')),
            SF('greatest', (any, List[any]), lambda s, p:
                self._pl_method_call(s, p, 'max')),
            SF('hash', (any, List[any]), lambda s, p:
               self._struct_apply_call(
                s, p,
                'lambda x: hash((v for v in x if))')),
            SF('least', (any, List[any]), lambda s, p:
                self._pl_method_call(s, p, 'min')),
            # SF('least', (any, List[any]), lambda s, p:
            #     self._struct_apply_call(
            #         s, p,
            #     'lambda x: np.min('
            #     '[v for v in x.values() if v is not None] or None)')),
            SF('lit', (any,), self._pl_method_call),
            SF('md5', (any, ), lambda s, p: self._series_apply_call(
                s, p,
                fn="lambda x: hashlib.sha1(x.encode('utf8')).hexdigest()"),
                ['import hashlib']),
            SF('nanvl', (any, any), self._nanvl_call),
            SF('sha1', (any, ), lambda s, p: self._series_apply_call(
                s, p,
                fn="lambda x: hashlib.sha1(x.encode('utf8')).hexdigest()"),
                ['import hashlib']),
            SF('sha2', (any, int), lambda s, p: self._series_apply_call(
                s, p,
                fn=f"lambda x: hashlib.sha{self.parse(s['arguments'][1], p)}"
                "(x.encode('utf8')).hexdigest()"),
                ['import hashlib']),
            SF('signum', (any, List[any]), self._signum_call),
            SF('to_json', (any, ),
               lambda s, p: self._series_apply_call(
                s, p, fn="to_json"),
                custom_functions={'to_json': to_json},
                imports=['import json']),

            # Array
            SF('array', (any, any),
                lambda s, p: self._pl_method_call(s, p, 'concat_list')),
            SF('array_contains', (any, any),
                lambda s, p: self._series_method_call(s, p, 'arr.contains')),
            SF('array_distinct', (any, any),
                lambda s, p: self._series_method_call(s, p, 'arr.unique')),
            SF('array_except', (any, any), self._array_set_call),
            SF('array_intersect', (any, any), self._array_set_call),
            SF('array_join', (any, any),
                lambda s, p: self._series_method_call(s, p, 'arr.join')),
            SF('array_max', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.max')),
            SF('array_min', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.min')),
            # https://github.com/pola-rs/polars/issues/5503
            SF('array_position', (any, any), lambda s, p: self._call_fmt(
                s, p,
                """(pl.when({0}.arr.contains({1})).then({0}.arr.eval(
                    (pl.element() == {1}).cast(pl.UInt8).arg_max() + 1
                ).arr.first()).otherwise(0))""")),

            SF('array_remove', (any, any), lambda s, p: self._call_fmt(
                s, p,
                "({0}.arr.eval(pl.element().filter(pl.element() != {1}})))")),
            SF('array_repeat', (any, any), lambda s, p:
                self._pl_method_call(s, p, 'repeat_by')),
            SF('array_sort', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.sort')),
            SF('array_union', (any, any), self._array_set_call),
            # SF('arrays_overlap', (any, any),
            #    lambda s, p: self._series_method_call(s, p, 'arr.max')),
            # SF('arrays_zip', (any, any),
            #    lambda s, p: self._series_method_call(s, p, 'arr.max')),
            SF('element_at', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.get')),
            SF('shuffle', (any, ), lambda s, p: self._series_apply_call(
                s, p,
                fn='lambda x: x.shuffle(random.randint(0, sys.maxsize))'),
                ['import random']),
            SF('size', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.lengths')),
            SF('slice', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.slice')),
            SF('sort_array', (any, ),
                lambda s, p: self._series_method_call(s, p, 'arr.sort')),

            # Data Explorer
            SF('array_cast', (any, ),
                lambda s, p: self._call_fmt(s, p, '{}.cast(pl.List(pl.{}))',
                                            set([1])),
               ),
            SF('isnotnull', (any, ),
                lambda s, p: self._series_method_call(s, p, 'is_not_null'),
               ),
            # SF('cast_array', (any, str), lambda s, p:
            #    (f"{self._arg(s, p, 0)}.arr"
            #     f".eval(pl.element().cast(pl.{self._arg(s, p, 1)}))")
            #   ),

            SF('extract_numbers', (any, ), lambda s, p:
                (f"{self._arg(s, p, 0)}"
                 f".str.extract_all(r'{number_regex}')"
                 f".arr.eval(pl.element().cast(pl.Float64))"
                 )
               ),

        ]

        # create_map',
        # 'explode', 'explode_outer',
        # 'posexplode',
        # 'flatten',
        # 'json_tuple',
        # 'map_concat', 'map_from_arrays', 'map_keys',
        # 'map_values',
        # 'sequence',
        # schema_of_json,
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
