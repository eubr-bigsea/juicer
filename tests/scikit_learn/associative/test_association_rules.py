from tests.scikit_learn import util
from juicer.scikit_learn.associative_operation import AssociationRulesOperation
from juicer.scikit_learn.library.rules_generator import \
    RulesGenerator
import pytest
from textwrap import dedent
import pandas as pd

# TODO: tests using/comparing with a well-known result


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# AssociationRules
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_association_rules_success():
    df = pd.DataFrame(
        [[[1], 0.363636], [[1, 5], 0.363636], [[1, 2], 0.363636],
         [[1, 2, 5], 0.363636],
         [[4], 0.363636], [[2, 4], 0.363636], [[3], 0.363636],
         [[2, 3], 0.363636], [[5], 0.363636], [[2, 3], 0.363636],
         [[5], 0.454545], [[2, 5], 0.454545], [[2], 0.545455]],
        columns=['itemsets', 'support'])
    test_df = df.copy()
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(test_df, col_item, col_freq)
    assert result['out'].equals(out)

    code = """
    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_rules_count_param_success():
    df = pd.DataFrame(
        [[[1], 0.363636], [[1, 5], 0.363636], [[1, 2], 0.363636],
         [[1, 2, 5], 0.363636],
         [[4], 0.363636], [[2, 4], 0.363636], [[3], 0.363636],
         [[2, 3], 0.363636], [[5], 0.363636], [[2, 3], 0.363636],
         [[5], 0.454545], [[2, 5], 0.454545], [[2], 0.545455]],
        columns=['itemsets', 'support'])
    test_df = df.copy()
    arguments = {
        'parameters': {'rules_count': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=3)
    out = rg.get_rules(test_df, col_item, col_freq)
    assert result['out'].equals(out)
    assert len(result['out']) == 3

    code = """
    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=3)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_confidence_param_success():
    df = pd.DataFrame(
        [[[1], 0.363636], [[1, 5], 0.363636], [[1, 2], 0.363636],
         [[1, 2, 5], 0.363636],
         [[4], 0.363636], [[2, 4], 0.363636], [[3], 0.363636],
         [[2, 3], 0.363636], [[5], 0.363636], [[2, 3], 0.363636],
         [[5], 0.454545], [[2, 5], 0.454545], [[2], 0.545455]],
        columns=['itemsets', 'support'])
    test_df = df.copy()
    arguments = {
        'parameters': {'confidence': 0.2},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.2, max_len=-1)
    out = rg.get_rules(test_df, col_item, col_freq)
    assert result['out'].equals(out)
    code = """
    col_item = "itemsets"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.2, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_attribute_param_success():
    df = pd.DataFrame(
        [[[1], 0.363636], [[1, 5], 0.363636], [[1, 2], 0.363636],
         [[1, 2, 5], 0.363636],
         [[4], 0.363636], [[2, 4], 0.363636], [[3], 0.363636],
         [[2, 3], 0.363636], [[5], 0.363636], [[2, 3], 0.363636],
         [[5], 0.454545], [[2, 5], 0.454545], [[2], 0.545455]],
        columns=['items', 'support'])
    test_df = df.copy()
    arguments = {
        'parameters': {'attribute': ["items"]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    col_item = "items"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(test_df, col_item, col_freq)
    assert result['out'].equals(out)

    code = """
    col_item = "items"
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_freq_param_success():
    df = pd.DataFrame(
        [[[1], 0.363636], [[1, 5], 0.363636], [[1, 2], 0.363636],
         [[1, 2, 5], 0.363636],
         [[4], 0.363636], [[2, 4], 0.363636], [[3], 0.363636],
         [[2, 3], 0.363636], [[5], 0.363636], [[2, 3], 0.363636],
         [[5], 0.454545], [[2, 5], 0.454545], [[2], 0.545455]],
        columns=['itemsets', 'support_2'])
    test_df = df.copy()
    arguments = {
        'parameters': {'freq': ['support_2']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    col_item = "itemsets"
    col_freq = "support_2"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(test_df, col_item, col_freq)
    assert result['out'].equals(out)
    code = """
    col_item = "itemsets"
    col_freq = "support_2"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = AssociationRulesOperation(**arguments)
    assert instance.generate_code() is None


def test_association_rules_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AssociationRulesOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_association_rules_invalid_confidence_param_fail():
    arguments = {
        'parameters': {'confidence': 2.0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        AssociationRulesOperation(**arguments)
    assert 'Confidence must be greater or' \
           ' equal to 0.0001 and smaller than 1.0' in str(val_err.value)
