from tests.scikit_learn import util
from juicer.scikit_learn.associative_operation import FrequentItemSetOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# FrequentItemSet
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_frequent_item_set_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == \
"""
col = 'df.columns[0]'
transactions = df[col].to_numpy().tolist()
dim = len(transactions)
min_support = 1.0 * dim

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
"""


def test_frequent_item_set_attribute_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1,
                       'attribute': ['support']},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == \
"""
col = ''support''
transactions = df[col].to_numpy().tolist()
dim = len(transactions)
min_support = 1.0 * dim

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
"""


def test_frequent_item_set_min_confidence_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'min_support': 1,
                       'min_confidence': 0.6},
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
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == \
"""
col = 'df.columns[0]'
transactions = df[col].to_numpy().tolist()
dim = len(transactions)
min_support = 1.0 * dim

patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
result = [[list(f), s] for f, s in patterns.items()]

col_item, col_freq = 'itemsets', 'support'

out = pd.DataFrame(result, columns=[col_item, col_freq])
out[col_freq] = out[col_freq] / dim
out = out.sort_values(by=col_freq, ascending=False)

# generating rules
from juicer.scikit_learn.library.rules_generator import RulesGenerator
rg = RulesGenerator(min_conf=0.6, max_len=-1)
rules_1 = rg.get_rules(out, col_item, col_freq)
"""


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
    assert "Parameter 'min_support' must be informed for task" in \
           str(val_err.value)


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