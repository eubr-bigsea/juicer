from tests.scikit_learn import util
from juicer.scikit_learn.associative_operation import AssociationRulesOperation
from juicer.scikit_learn.library.rules_generator import \
    RulesGenerator
import pytest
from textwrap import dedent
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# AssociationRules
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_association_rules_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
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
    assert not result['out'].equals(test_df)

    code = """
    col_item = df.columns[0]
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_rules_count_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
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
    assert not result['out'].equals(test_df)

    code = """
    col_item = df.columns[0]
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=3)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_confidence_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'confidence': 0.8},
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
    assert not result['out'].equals(test_df)

    code = """
    col_item = df.columns[0]
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.8, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_attribute_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'attribute': ["'sepallength'"]},
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
    assert not result['out'].equals(test_df)

    code = """
    col_item = 'sepallength'
    col_freq = "support"

    rg = RulesGenerator(min_conf=0.5, max_len=-1)
    out = rg.get_rules(df, col_item, col_freq)   
    """

    assert dedent(code) == instance.generate_code()


def test_association_rules_freq_param_success():
    one = [f'str{i}' for i in range(10)]
    data = {'sepallength': one,
            'support': one}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'freq': ['sepallength']},
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
    assert not result['out'].equals(test_df)

    code = """
    col_item = df.columns[0]
    col_freq = "sepallength"

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

