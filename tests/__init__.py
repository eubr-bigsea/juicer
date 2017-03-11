# -*- coding: utf-8 -*-
import ast


def format_code_comparison(node1, node2):
    return '\n>>>>>\n{}\n>>>>>\n{}\n-----'.format(node1, node2)


def compare_ast(node1, node2):
    if type(node1) != type(node2):
        return False, 'Different types: ({}) != ({}) [{}, {}]'.format(
            type(node1), type(node2), node1, node2)
    elif isinstance(node1, ast.AST):
        for kind, var in vars(node1).items():
            if kind not in ('lineno', 'col_offset', 'ctx'):
                var2 = vars(node2).get(kind)
                result, msg = compare_ast(var, var2)
                if not result:
                    return False, msg
        return True, ''
    elif isinstance(node1, list):
        if len(node1) != len(node2):
            out1 = node1
            out2 = node2
            resp = ['Different lenght in nodes ({}, {})'.format(len(out1), len(out2))]
            resp.append(', '.join([x.__class__.__name__ for x in out1]))
            resp.append(', '.join([x.__class__.__name__ for x in out2]))
            return False, '\n'.join(resp)
        for i in range(len(node1)):
            result, msg = compare_ast(node1[i], node2[i])
            if not result:
                return False, msg
        return True, ''
    else:
        return node1 == node2, 'Node comparison ({}) == ({})'.format(node1,
                                                                     node2)
