from tests.scikit_learn import util
from juicer.scikit_learn.text_operation import WordToVectorOperation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def do_nothing(tokens):
    return tokens


# WordToVector
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_word_to_vector_type_count_vocab_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 3},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    out = test_df

    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = CountVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1, max_features=3)

    vector_model_1.fit(corpus)
    out['class_vec'] = vector_model_1.transform(corpus).toarray().tolist()

    assert result['out'].equals(out)
    assert len(result['out'].iloc[0, 1]) == 3


def test_word_to_vector_type_count_minimum_df_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'minimum_df': 0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df

    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = CountVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1.0, max_features=1000)

    vector_model_1.fit(corpus)
    out['class_vec'] = vector_model_1.transform(corpus).toarray().tolist()

    assert result['out'].equals(out)


def test_word_to_vector_type_count_alias_param_success():
    df = util.iris(['class'], size=150)

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
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[1] == 'success'


def test_word_to_vector_type_count_all_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 3, 'minimum_df': 0,
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df

    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = CountVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1.0, max_features=3)

    vector_model_1.fit(corpus)
    out['success'] = vector_model_1.transform(corpus).toarray().tolist()
    assert result['out'].equals(out)
    assert result['out'].columns[1] == 'success'
    assert len(result['out'].iloc[0, 1]) == 3


def test_word_to_vector_type_tfidf_vocab_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 2, 'type': 'TF-IDF'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df

    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = TfidfVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1, max_features=2)

    vector_model_1.fit(corpus)
    out['class_vec'] = vector_model_1.transform(corpus).toarray().tolist()
    assert result['out'].equals(out)
    assert len(result['out'].iloc[0, 1]) == 2


def test_word_to_vector_type_tfidf_minimum_df_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'minimum_df': 1, 'type': 'TF-IDF'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df

    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = TfidfVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1, max_features=1000)

    vector_model_1.fit(corpus)
    out['class_vec'] = vector_model_1.transform(corpus).toarray().tolist()

    assert result['out'].equals(out)


def test_word_to_vector_type_tfidf_alias_param_success():
    df = util.iris(['class'], size=150)

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success', 'type': 'TF-IDF'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[1] == 'success'


def test_word_to_vector_type_tfidf_all_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 2, 'minimum_df': 1, 'alias': 'success',
                       'type': 'TF-IDF'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df
    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = TfidfVectorizer(tokenizer=do_nothing,
                                     preprocessor=None, lowercase=False,
                                     min_df=1, max_features=2)

    vector_model_1.fit(corpus)
    out['success'] = vector_model_1.transform(corpus).toarray().tolist()
    assert result['out'].equals(out)
    assert len(result['out'].iloc[0, 1]) == 2
    assert result['out'].columns[1] == 'success'


def test_word_to_vector_type_word2vec_vocab_param_success():
    df = util.iris(['class'], size=150)
    df['class'] = df['class'].apply(lambda row: row.split("-"))
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 3, 'type': 'word2vec'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'vector_model_X'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['vector_model_X'].equals(test_df)
    assert len(result['vector_model_X'].iloc[0, 1]) == 3


def test_word_to_vector_type_word2vec_minimum_df_param_success():
    df = util.iris(['class'], size=10)
    df['class'] = df['class'].apply(lambda row: row.split("-"))
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}, 'type': 'word2vec',
                       'minimum_df': 5},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'vector_model_X'
        }
    }

    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['vector_model_X'].equals(test_df)


def test_word_to_vector_type_word2vec_alias_param_success():
    df = util.iris(['class'], size=10)
    df['class'] = df['class'].apply(lambda row: row.split("-"))
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}, 'type': 'word2vec',
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'vector_model_X'
        }
    }

    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['vector_model_X'].columns[1] == 'success'


def test_word_to_vector_type_word2vec_all_param_success():
    df = util.iris(['class'], size=10)
    df['class'] = df['class'].apply(lambda row: row.split("-"))
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}, 'type': 'word2vec',
                       'vocab_size': 3,
                       'minimum_df': 5,
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'vector_model_X'
        }
    }

    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['vector_model_X'].equals(test_df)
    assert result['vector_model_X'].columns[1] == 'success'
    assert len(result['vector_model_X'].iloc[0, 1]) == 3


def test_word_to_vector_type_hashing_tf_vocab_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 5, 'type': 'hashing_tf'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df
    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = HashingVectorizer(tokenizer=do_nothing,
                                       preprocessor=None, lowercase=False,
                                       n_features=5)
    vector_model_1.fit(corpus)

    vector = vector_model_1.transform(corpus).toarray().tolist()
    out['class_vec'] = vector
    assert result['out'].equals(out)
    assert len(result['out'].iloc[0, 1]) == 5


def test_word_to_vector_type_hashing_tf_minimum_df_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'minimum_df': 1, 'type': 'hashing_tf'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df
    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = HashingVectorizer(tokenizer=do_nothing,
                                       preprocessor=None, lowercase=False,
                                       n_features=1000)
    vector_model_1.fit(corpus)

    vector = vector_model_1.transform(corpus).toarray().tolist()
    out['class_vec'] = vector
    assert result['out'].equals(out)


def test_word_to_vector_type_hashing_tf_alias_param_success():
    df = util.iris(['class'], size=150)

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success', 'type': 'hashing_tf'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[1] == 'success'


def test_word_to_vector_type_hashing_tf_all_param_success():
    df = util.iris(['class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'vocab_size': 5, 'minimum_df': 1, 'alias': 'success',
                       'type': 'hashing_tf'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    out = test_df
    corpus = out['class'].to_numpy().tolist()
    vector_model_1 = HashingVectorizer(tokenizer=do_nothing,
                                       preprocessor=None, lowercase=False,
                                       n_features=5)
    vector_model_1.fit(corpus)

    vector = vector_model_1.transform(corpus).toarray().tolist()
    out['success'] = vector

    assert result['out'].equals(out)
    assert result['out'].columns[1] == 'success'
    assert len(result['out'].iloc[0, 1]) == 5


def test_word_to_vector_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = WordToVectorOperation(**arguments)
    assert instance.generate_code() is None


def test_word_to_vector_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_word_to_vector_missing_attributes_param_fail():
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
        WordToVectorOperation(**arguments)
    assert "Parameter 'attributes' must be informed for task" in \
           str(val_err.value)


def test_word_to_vector_invalid_type_param_fail():
    arguments = {
        'parameters': {'attributes': ['class'],
                       'multiplicity': {'input data': 0},
                       'type': 'invalid'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        WordToVectorOperation(**arguments)
    assert "Invalid type 'invalid' for task" in str(val_err.value)


def test_word_to_vector_missing_multiplicity_param_fail():
    df = util.iris(['class'], size=150)
    arguments = {
        'parameters': {'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = WordToVectorOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "multiplicity" in str(key_err.value)
