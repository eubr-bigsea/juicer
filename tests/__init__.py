# -*- coding: utf-8 -*-
from __future__ import absolute_import

import ast
import gettext
from itertools import zip_longest
import os
from textwrap import dedent

locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
t = gettext.translation('messages', locales_path, ["pt"],
                        fallback=True)
t.install()


def format_code_comparison(node1, node2, msg=None):
    lines_left = node1.split('\n')
    lines_right = node2.split('\n')

    code = []
    if msg:
        code.append(msg)
    code.extend(['{:<4} {:<85} | {}'.format(i + 1, l[:81] if l else '',
                                            r[:81] if r else '')
                 for i, (l, r) in enumerate(
                     zip_longest(lines_left, lines_right))])
    return '\n' + '\n'.join(code)


current_line = current_offset = 0


def compare_ast(node1, node2):
    global current_offset, current_line
    if type(node1) != type(node2):
        return (False,
                f'Different types: ({type(node1)}) != ({type(node2)}) '
                f'[{node1}, {node2}]')
    elif isinstance(node1, ast.AST):
        for kind, var in vars(node1).items():
            if kind == 'lineno':
                current_line = vars(node2)['lineno']
            elif kind == 'col_offset':
                current_offset = vars(node2)['col_offset']
            elif kind not in ('ctx',):
                var2 = vars(node2).get(kind)
                result, msg = compare_ast(var, var2)
                if not result:
                    return False, f'{current_line}:{current_offset}] {msg}'
        return True, ''
    elif isinstance(node1, list):
        if len(node1) != len(node2):
            out1 = node1
            out2 = node2
            resp = ['Different lenght in nodes ({}, {})'.format(
                len(out1), len(out2)),
                ', '.join([x.__class__.__name__ for x in out1]),
                ', '.join([x.__class__.__name__ for x in out2])]
            return False, '\n'.join(resp)
        for i in range(len(node1)):
            result, msg = compare_ast(node1[i], node2[i])
            if not result:
                return False, msg
        return True, ''
    else:
        return node1 == node2, f'Node comparison ({node1}) == ({node2})'


def debug_ast(code, expected_code):
    sep = '-'
    print(dedent(f"""
    Code
    {sep}
    {code}
    {sep}
    Expected
    {sep}
    {expected_code}
    """))
