# -*- coding: utf-8 -*-
import ast


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
            return False, 'Different lenght in nodes: {} {}'.format(out1, out2)
        for i in range(len(node1)):
            result, msg = compare_ast(node1[i], node2[i])
            if not result:
                return False, msg
        return True, ''
    else:
        return node1 == node2, 'Node comparison ({}) == ({})'.format(node1,
                                                                     node2)
