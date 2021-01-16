from tests.scikit_learn import util
from juicer.scikit_learn.text_operation import RemoveStopWordsOperation
from nltk.corpus import stopwords
import pytest
import pandas as pd
import nltk


# nltk.download('stopwords')

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# RemoveStopWords
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_remove_stop_words_success():
    df = util.iris(['class'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    stop_words = []
    stop_words = [w.lower() for w in stop_words]
    word_tokens = test_out['class'].to_numpy()
    test_result = []
    for row in word_tokens:
        itr = []
        for w in row:
            if not w.lower() in stop_words:
                itr.append(w)
        test_result.append(itr)
    test_out['tokenized_rm'] = test_result

    assert result['out'].equals(test_out)


def test_remove_stop_words_alias_param_success():
    df = util.iris(['class'], size=10)
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    assert result['out'].columns[1] == 'success'


def test_remove_stop_words_stop_word_list_param_success():
    df = util.iris(['class'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'stop_word_list': 'Do,does,did'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    stop_words = ['Do', 'does', 'did']
    stop_words = [w.lower() for w in stop_words]
    word_tokens = test_out['class'].to_numpy()
    test_result = []
    for row in word_tokens:
        itr = []
        for w in row:
            if not w.lower() in stop_words:
                itr.append(w)
        test_result.append(itr)
    test_out['tokenized_rm'] = test_result

    assert result['out'].equals(test_out)


def test_remove_stop_words_stop_word_attribute_param_success():
    df = util.titanic(['name', 'homedest'], size=10)
    test_df = df.copy()

    data = {'w_attribute': ['Do', 'does', 'did']}
    stp_w = pd.DataFrame(data)

    arguments = {
        'parameters': {'attributes': ['name'],
                       'multiplicity': {'input data': 0},
                       'stop_word_attribute': ['w_attribute']},
        'named_inputs': {
            'input data': 'df',
            'stop words': 'stp_w'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df,
                           'stp_w': stp_w})

    test_out = test_df
    stop_words = []
    stop_words += stp_w['w_attribute'].to_numpy().tolist()

    stop_words = [w.lower() for w in stop_words]
    word_tokens = test_out['name'].to_numpy()
    test_result = []
    for row in word_tokens:
        itr = []
        for w in row:
            if not w.lower() in stop_words:
                itr.append(w)
        test_result.append(itr)
    test_out['tokenized_rm'] = test_result

    assert result['out'].equals(test_out)


def test_remove_stop_words_stop_word_language_param_success():
    df = util.iris(['class'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'stop_word_language': 'asdas',
                       'language': 'portuguese'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    stop_words = []
    stop_words += stopwords.words('portuguese')
    stop_words = [w.lower() for w in stop_words]
    word_tokens = test_out['class'].to_numpy()
    test_result = []
    for row in word_tokens:
        itr = []
        for w in row:
            if not w.lower() in stop_words:
                itr.append(w)
        test_result.append(itr)
    test_out['tokenized_rm'] = test_result

    assert result['out'].equals(test_out)


def test_remove_stop_words_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    assert instance.generate_code() is None


def test_remove_stop_words_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_remove_stop_words_missing_attributes_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RemoveStopWordsOperation(**arguments)
    assert "Parameter 'attributes' must be informed for task" \
           in str(val_err.value)


def test_remove_stop_words_missing_multiplicity_param_fail():
    df = util.iris(['class'], size=10)
    arguments = {
        'parameters': {'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RemoveStopWordsOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert 'multiplicity' in str(key_err.value)
