from tests.scikit_learn import util
from juicer.scikit_learn.associative_operation import FrequentItemSetOperation
from juicer.scikit_learn.library.rules_generator import RulesGenerator
from textwrap import dedent
import pytest
import pandas as pd
import pyfpgrowth


# TODO: tests using/comparing with a well-known result
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# FrequentItemSet
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_frequent_item_set_success():
    transactions = [[[1, 2, 5], 1],
                    [[2, 4], 2],
                    [[2, 3], 3],
                    [[1, 2, 4], 4],
                    [[1, 3], 5],
                    [[2, 3], 6],
                    [[1, 3], 7],
                    [[1, 2, 3, 5], 8],
                    [[1, 2, 3], 9]]

    df = pd.DataFrame(transactions, columns=['transactions', 'id'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.222},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FrequentItemSetOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    transactions = test_df['transactions'].to_numpy().tolist()
    dim = float(len(transactions))
    min_support = 0.222 * dim
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    oper_result = [[list(f), s] for f, s in patterns.items()]
    col_item, col_freq = 'itemsets', 'support'
    out = pd.DataFrame(oper_result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out.sort_values(by=col_freq, ascending=False, inplace=True)
    rg = RulesGenerator(min_conf=0.9, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)

    assert result['rules_1'].equals(rules_1)
    assert result['out'].equals(out)

    assert dedent("""
    col = 'df.columns[0]'
    transactions = df[col].to_numpy().tolist()
    dim = len(transactions)
    min_support = 0.222 * dim
    
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    result = [[list(f), s] for f, s in patterns.items()]
    
    col_item, col_freq = 'itemsets', 'support'
    
    out = pd.DataFrame(result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out = out.sort_values(by=col_freq, ascending=False)
    
    # generating rules
    from juicer.scikit_learn.library.rules_generator import RulesGenerator
    rg = RulesGenerator(min_conf=0.9, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)
    """) == instance.generate_code()


def test_frequent_item_set_attribute_and_min_support_params_success():
    transactions = [[[1, 2, 5], 1],
                    [[2, 4], 2],
                    [[2, 3], 3],
                    [[1, 2, 4], 4],
                    [[1, 3], 5],
                    [[2, 3], 6],
                    [[1, 3], 7],
                    [[1, 2, 3, 5], 8],
                    [[1, 2, 3], 9]]

    df = pd.DataFrame(transactions, columns=['transactions', 'id'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.5,
                       'attribute': ['transactions']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FrequentItemSetOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    transactions = test_df['transactions'].to_numpy().tolist()
    dim = float(len(transactions))
    patterns = pyfpgrowth.find_frequent_patterns(transactions, (dim * 0.5))
    oper_result = [[list(f), s] for f, s in patterns.items()]
    col_item, col_freq = 'itemsets', 'support'
    out = pd.DataFrame(oper_result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out.sort_values(by=col_freq, ascending=False, inplace=True)
    rg = RulesGenerator(min_conf=0.9, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)

    assert result['rules_1'].equals(rules_1)
    assert result['out'].equals(out)

    assert dedent("""
    col = ''transactions''
    transactions = df[col].to_numpy().tolist()
    dim = len(transactions)
    min_support = 0.5 * dim
               
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    result = [[list(f), s] for f, s in patterns.items()]
               
    col_item, col_freq = 'itemsets', 'support'
               
    out = pd.DataFrame(result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out = out.sort_values(by=col_freq, ascending=False)
               
    # generating rules
    from juicer.scikit_learn.library.rules_generator import RulesGenerator
    rg = RulesGenerator(min_conf=0.9, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)
    """) == instance.generate_code()


def test_frequent_item_set_min_confidence_param_success():
    transactions = [[[1, 2, 5], 1],
                    [[2, 4], 2],
                    [[2, 3], 3],
                    [[1, 2, 4], 4],
                    [[1, 3], 5],
                    [[2, 3], 6],
                    [[1, 3], 7],
                    [[1, 2, 3, 5], 8],
                    [[1, 2, 3], 9]]

    df = pd.DataFrame(transactions, columns=['transactions', 'id'])
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 0.222,
                       'min_confidence': 0.5},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FrequentItemSetOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    transactions = test_df[test_df.columns[0]].to_numpy().tolist()
    dim = float(len(transactions))
    patterns = pyfpgrowth.find_frequent_patterns(transactions, dim * 0.222)
    oper_result = [[list(f), s] for f, s in patterns.items()]
    col_item, col_freq = 'itemsets', 'support'
    out = pd.DataFrame(oper_result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out.sort_values(by=col_freq, ascending=False, inplace=True)
    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)

    assert result['rules_1'].equals(rules_1)
    assert result['out'].equals(out)

    assert dedent("""
    col = 'df.columns[0]'
    transactions = df[col].to_numpy().tolist()
    dim = len(transactions)
    min_support = 0.222 * dim
           
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
    result = [[list(f), s] for f, s in patterns.items()]
           
    col_item, col_freq = 'itemsets', 'support'
           
    out = pd.DataFrame(result, columns=[col_item, col_freq])
    out[col_freq] = out[col_freq] / dim
    out = out.sort_values(by=col_freq, ascending=False)
    
    # generating rules
    from juicer.scikit_learn.library.rules_generator import RulesGenerator
    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    rules_1 = rg.get_rules(out, col_item, col_freq)
    """) == instance.generate_code()


def test_frequent_item_set_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'min_support': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = FrequentItemSetOperation(**arguments)
    assert instance.generate_code() is None


def test_frequent_item_set_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'min_support': 1},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FrequentItemSetOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_frequent_item_set_missing_min_support_param_fail():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        FrequentItemSetOperation(**arguments)
    assert f"Parameter 'min_support' must be informed for task " \
           f"{FrequentItemSetOperation}" in str(val_err.value)


def test_frequent_item_set_invalid_min_support_param_fail():
    arguments = {
        'parameters': {'min_support': -1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        FrequentItemSetOperation(**arguments)
    assert "Support must be greater or equal to 0.0001 and smaller than 1.0" in \
           str(val_err.value)
