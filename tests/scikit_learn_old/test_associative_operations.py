# -*- coding: utf-8 -*-
import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.scikit_learn.associative_operation import \
    FrequentItemSetOperation, \
    SequenceMiningOperation, \
    AssociationRulesOperation

from tests import compare_ast, format_code_comparison

"""
    Frequent ItemSet Operation Tests
"""


def test_frequent_itemset_minimum_success():
    params = {
        FrequentItemSetOperation.MIN_SUPPORT_PARAM: 0.5
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = FrequentItemSetOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        col = input_1.columns[0]
        transactions = input_1[col].values.tolist()
        min_support = 100 * 0.5
        
        result = fpgrowth(transactions, target="s",
          supp=min_support, report="s")
         
        output_1 = pd.DataFrame(result, columns=['itemsets', 'support'])
         
        rules_1 = None
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_frequent_itemset_success():
    params = {
        FrequentItemSetOperation.ATTRIBUTE_PARAM: ['col_2'],
        FrequentItemSetOperation.MIN_SUPPORT_PARAM: 0.7

    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_0', 'rules output': 'df_rules'}

    instance = FrequentItemSetOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        col = 'col_2'
        transactions = input_1[col].values.tolist()
        min_support = 100 * 0.7
        
        result = fpgrowth(transactions, target="s",
          supp=min_support, report="s")
         
        output_0 = pd.DataFrame(result, columns=['itemsets', 'support'])
         
        # generating rules
        col_item = 'itemsets'
        col_freq = 'support'
        rg = RulesGenerator(min_conf=0.9, max_len=-1)
        df_rules = rg.get_rules(output_0, col_item, col_freq) 
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_frequent_itemset_rules_success():
    params = {
        FrequentItemSetOperation.ATTRIBUTE_PARAM: ['col_2'],
        FrequentItemSetOperation.MIN_SUPPORT_PARAM: 0.8

    }
    n_in = {'input data': 'input_1'}
    n_out = {'rules output': 'df_rules'}

    instance = FrequentItemSetOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         col = 'col_2'
         transactions = input_1[col].values.tolist()
         min_support = 100 * 0.8
         
         result = fpgrowth(transactions, target="s",
           supp=min_support, report="s")
         
         output_data_1 = pd.DataFrame(result, columns=['itemsets', 'support'])
         
         
         # generating rules
         col_item = 'itemsets'
         col_freq = 'support'
         rg = RulesGenerator(min_conf=0.9, max_len=-1)
         df_rules = rg.get_rules(output_data_1, col_item, col_freq)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_frequent_itemset_model_operation_missing_features_failure():
    params = {FrequentItemSetOperation.MIN_SUPPORT_PARAM: -0.5}
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        FrequentItemSetOperation(params, named_inputs=n_in, named_outputs=n_out)


"""
    Sequence Mining Tests
"""


def test_sequence_mining_minimum_success():
    params = {
        SequenceMiningOperation.MIN_SUPPORT_PARAM: 0.5,
        SequenceMiningOperation.MAX_LENGTH_PARAM: 10,

    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = SequenceMiningOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         col = input_1.columns[0]
         transactions = input_1[col].values.tolist()
         min_support = 0.5
         max_length = 10
         
         span = PrefixSpan(transactions)
         span.run(min_support, max_length)
         result = span.get_patest_text_operations.pyatterns()
         
         output_1 = pd.DataFrame(result, columns=['itemsets', 'support'])
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_sequence_mining_success():
    params = {
        SequenceMiningOperation.MIN_SUPPORT_PARAM: 0.8,
        SequenceMiningOperation.MAX_LENGTH_PARAM: 11,
        SequenceMiningOperation.ATTRIBUTE_PARAM: ['col_3'],

    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_2'}

    instance = SequenceMiningOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         col = 'col_3'
         transactions = input_1[col].values.tolist()
         min_support = 0.8
         max_length = 11

         span = PrefixSpan(transactions)
         span.run(min_support, max_length)
         result = span.get_patest_text_operations.pyatterns()

         output_2 = pd.DataFrame(result, columns=['itemsets', 'support'])
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_sequence_mining_missing_features_failure():
    params = {}
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        SequenceMiningOperation(params, named_inputs=n_in, named_outputs=n_out)


"""
    Association Rules Operation tests
"""


def test_association_rules_minimum_success():
    params = {
        AssociationRulesOperation.CONFIDENCE_PARAM: 0.9,
        AssociationRulesOperation.MAX_COUNT_PARAM: 280

    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_2'}

    instance = AssociationRulesOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        col_item = input_1.columns[0]
        col_freq = "support"
         
        rg = RulesGenerator(min_conf=0.9, max_len=280)
        output_2 = rg.get_rules(input_1, col_item, col_freq)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_association_rules_success():
    params = {
        AssociationRulesOperation.CONFIDENCE_PARAM: 0.3,
        AssociationRulesOperation.MAX_COUNT_PARAM: 281,
        AssociationRulesOperation.ITEMSET_ATTR_PARAM: ['col_3']

    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_2'}

    instance = AssociationRulesOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        col_item = 'col_3'
        col_freq = "support"

        rg = RulesGenerator(min_conf=0.3, max_len=281)
        output_2 = rg.get_rules(input_1, col_item, col_freq)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)