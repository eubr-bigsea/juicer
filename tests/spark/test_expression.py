# coding=utf-8
from __future__ import absolute_import

import ast

import pytest
from juicer.spark.expression import Expression
from tests import compare_ast, format_code_comparison


def test_unary_expression_valid_success():
    json_code = {
        "type": "UnaryExpression",
        "operator": "~",
        "argument": {
            "type": "Identifier",
            "name": "column"
        },
        "prefix": True
    }
    params = {}

    expr = Expression(json_code, params)

    code = expr.parsed_expression
    expected_code = "(~ functions.col('column'))"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)

    json_code['operator'] = '-'
    json_code['argument']['type'] = 'Literal'
    json_code['argument']['value'] = 1
    json_code['argument']['raw'] = '1'
    expr = Expression(json_code, params)

    result, msg = compare_ast(ast.parse(expr.parsed_expression),
                              ast.parse("-1"))
    assert result, msg

    json_code['operator'] = '+'
    json_code['argument']['type'] = 'Literal'
    json_code['argument']['value'] = 'some text'
    json_code['argument']['raw'] = "'some text'"
    expr = Expression(json_code, params)

    code = expr.parsed_expression
    expected_code = "+ 'some text'"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_binary_expression_valid_success():
    json_code = {
        "type": "BinaryExpression",
        "operator": "*",
        "left": {
            "type": "Identifier",
            "name": "column1"
        },
        "right": {
            "type": "Identifier",
            "name": "column2"
        }
    }
    params = {}
    expr = Expression(json_code, params)
    result, msg = compare_ast(ast.parse(expr.parsed_expression),
                              ast.parse(
                                  "functions.col('column1') * functions.col('column2')"))
    assert result, msg

    json_code['operator'] = '/'
    json_code['left']['type'] = 'Literal'
    json_code['left']['value'] = 100
    json_code['left']['raw'] = '100'
    expr = Expression(json_code, params)

    result, msg = compare_ast(ast.parse(expr.parsed_expression),
                              ast.parse("100 / functions.col('column2')"))
    assert result, msg


def test_binary_expression_with_params_success():
    json_code = {
        "type": "BinaryExpression",
        "operator": "*",
        "left": {
            "type": "Identifier",
            "name": "column1"
        },
        "right": {
            "type": "Identifier",
            "name": "column2"
        }
    }
    params = {
        'input': 'df00'
    }
    expr = Expression(json_code, params)
    result, msg = compare_ast(ast.parse(expr.parsed_expression),
                              ast.parse("df00['column1'] * df00['column2']"))
    assert result, msg

    json_code['operator'] = '/'
    json_code['left']['type'] = 'Literal'
    json_code['left']['value'] = 100
    json_code['left']['raw'] = '100'
    expr = Expression(json_code, params)

    result, msg = compare_ast(ast.parse(expr.parsed_expression),
                              ast.parse("100 / df00['column2']"))
    assert result, msg


def test_binary_call_expression_with_params_success():
    json_code = {
        "type": "CallExpression",
        "arguments": [
            {
                "type": "BinaryExpression",
                "operator": "*",
                "left": {
                    "type": "Identifier",
                    "name": "column1"
                },
                "right": {
                    "type": "Identifier",
                    "name": "column2"
                }
            },
            {
                "type": "Literal",
                "value": 20,
                "raw": "20"
            }
        ],
        "callee": {
            "type": "Identifier",
            "name": "pow"
        }
    }
    params = {
        'input': 'df00'
    }
    expr = Expression(json_code, params)
    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        "functions.pow(df00['column1'] * df00['column2'], 20)"))
    assert result, msg

    json_code['operator'] = '/'
    json_code['arguments'][0]['type'] = 'Literal'
    json_code['arguments'][0]['value'] = 100
    json_code['arguments'][0]['raw'] = '100'
    expr = Expression(json_code, params)

    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        "functions.pow(100, 20)"))
    assert result, msg


def test_logical_expression_success():
    json_code = {
        "type": "LogicalExpression",
        "operator": "&&",
        "left": {
            "type": "LogicalExpression",
            "operator": "||",
            "left": {
                "type": "Identifier",
                "name": "a"
            },
            "right": {
                "type": "Identifier",
                "name": "b"
            }
        },
        "right": {
            "type": "UnaryExpression",
            "operator": "!",
            "argument": {
                "type": "Identifier",
                "name": "c"
            },
            "prefix": True
        }
    }
    params = {
        'input': 'df00'
    }
    expr = Expression(json_code, params)
    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        "(df00['a'] | df00['b']) & ~df00['c']"))
    assert result, msg + expr.parsed_expression

    json_code['operator'] = '||'
    json_code['left']['type'] = 'Literal'
    json_code['left']['value'] = 100
    json_code['left']['raw'] = '100'
    expr = Expression(json_code, params)

    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        "(100) | ~ (df00['c'])"))
    assert result, msg + expr.parsed_expression


def test_conditional_expression_success():
    json_code = {
        "type": "ConditionalExpression",
        "test": {
            "type": "BinaryExpression",
            "operator": ">",
            "left": {
                "type": "Identifier",
                "name": "a"
            },
            "right": {
                "type": "Literal",
                "value": 1,
                "raw": "1"
            }
        },
        "consequent": {
            "type": "Literal",
            "value": 2,
            "raw": "2"
        },
        "alternate": {
            "type": "Literal",
            "value": 3,
            "raw": "3"
        }
    }
    params = {
        'input': 'df00'
    }
    expr = Expression(json_code, params)
    expected_code = "functions.when((df00['a'] > 1), 2).otherwise(3)"
    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        expected_code))
    assert result, msg + format_code_comparison(expr.parsed_expression,
                                                expected_code)

    json_code['consequent']['type'] = 'Identifier'
    json_code['consequent']['value'] = 'ok'
    json_code['consequent']['name'] = 'ok'
    json_code['consequent']['raw'] = '"ok"'
    expr = Expression(json_code, params)

    expected_code = "functions.when(df00['a'] > 1, df00['ok']).otherwise(3)"
    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        expected_code))
    assert result, msg + format_code_comparison(expr.parsed_expression,
                                                expected_code)


def test_unknown_type_expression_failure():
    with pytest.raises(ValueError):
        json_code = {
            "type": "InvalidBinaryExpression",
            "operator": ">",
            "left": {
                "type": "Identifier",
                "name": "a"
            },
            "right": {
                "type": "Literal",
                "value": 1,
                "raw": "1"
            }
        }
        Expression(json_code, {})


def test_get_windows_function_success():
    json_code = {
        "type": "CallExpression",
        "arguments": [
            {
                "type": "Identifier",
                "name": "created_at"
            },
            {
                "type": "Literal",
                "value": 10,
                "raw": "10"
            },
            {
                "type": "Literal",
                "value": "end",
                "raw": "'end'"
            }
        ],
        "callee": {
            "type": "Identifier",
            "name": "window"
        }
    }
    params = {}

    expr = Expression(json_code, params, True)

    expected_code = ("functions.window("
                     "functions.col('created_at'),"
                     "str('10 seconds')).start.cast('timestamp')")
    result, msg = compare_ast(ast.parse(expr.parsed_expression), ast.parse(
        expected_code))
    assert result, msg + format_code_comparison(expr.parsed_expression,
                                                expected_code)
