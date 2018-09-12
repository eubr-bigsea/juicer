# -*- coding: utf-8 -*-
import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.scikit_learn.text_operation import TokenizerOperation, \
    RemoveStopWordsOperation, \
    WordToVectorOperation, GenerateNGramsOperation

from tests import compare_ast, format_code_comparison

"""
    Tokenizer Operation Tests
"""


def test_tokenizer_minimum_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'simple',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = TokenizerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         result = []
         toktok = ToktokTokenizer()
         
         for row in output_1['col'].values:
             result.append([word 
             for word in toktok.tokenize(row) if len(word) >= 3])
         output_1['col_tok'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_tokenizer_operation_type_simple_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'simple',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col_0'],
        TokenizerOperation.ALIAS_PARAM: 'col_1',
        TokenizerOperation.MINIMUM_SIZE: 5,
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = TokenizerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         result = []
         toktok = ToktokTokenizer()
         
         for row in output_1['col_0'].values:
             result.append([word 
             for word in toktok.tokenize(row) if len(word) >= 5])
         output_1['col_1'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_tokenizer_operation_type_regexp_success():
    params = {
        TokenizerOperation.TYPE_PARAM: 'regex',
        TokenizerOperation.ATTRIBUTES_PARAM: ['col'],
        TokenizerOperation.ALIAS_PARAM: 'col_2',
        TokenizerOperation.MINIMUM_SIZE: 4,
        TokenizerOperation.EXPRESSION_PARAM: r'\s+'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = TokenizerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         result = []
         
         for row in output_1['col'].values:
             result.append([word 
             for word in regexp_tokenize(row, pattern='\s+') if len(word) >= 4])
         output_1['col_2'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_tokenizer_model_operation_missing_features_failure():
    params = {}
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        TokenizerOperation(params, named_inputs=n_in, named_outputs=n_out)


"""
    Remove Stopwords Operation Tests
"""


def test_remove_stopwords_operations_2_params_success():

    n_in = {'input data': 'input_1', 'stop words': 'df_stops'}
    n_out = {'output data': 'output_1'}

    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: ['text'],
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: n_in['stop words'],
        RemoveStopWordsOperation.STOP_WORD_ATTRIBUTE_PARAM: ['stop_word'],
        RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM: True
    }
    # Input data, and StopWords list

    instance = RemoveStopWordsOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
         stop_words = []
         output_1 = input_1.copy()
         
         stop_words += df_stops['stop_word'].values.tolist()
         
         word_tokens = output_1['text'].values       
         result = []
         for row in word_tokens:
             result.append([w for w in row if not w in stop_words])
         output_1['tokenized_rm'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_stopwords_operations_2_false_params_success():
    n_in = {'input data': 'input_1', 'stop words': 'df_stops'}
    n_out = {'output data': 'output_1'}

    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: ['text'],
        RemoveStopWordsOperation.ALIAS_PARAM: 'col_out',
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: n_in['stop words'],
        RemoveStopWordsOperation.STOP_WORD_ATTRIBUTE_PARAM: ['stop_word'],
        RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM: False
    }
    # Input data, and StopWords list

    instance = RemoveStopWordsOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
         stop_words = []
         output_1 = input_1.copy()
         
         stop_words += df_stops['stop_word'].values.tolist()
         
         stop_words = [w.lower() for w in stop_words]
         word_tokens = output_1['text'].values       
         result = []
         for row in word_tokens:
             result.append([w for w in row if not w.lower() in stop_words])
         output_1['col_out'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_stopwords_operations_1_params_success():
    params = {
        RemoveStopWordsOperation.ATTRIBUTES_PARAM: ['text2'],
        RemoveStopWordsOperation.ALIAS_PARAM: 'col3',
        RemoveStopWordsOperation.STOP_WORD_LIST_PARAM: 'stop_word_list',
        RemoveStopWordsOperation.STOP_WORD_LANGUAGE_PARAM: 'english',
        RemoveStopWordsOperation.STOP_WORD_CASE_SENSITIVE_PARAM: 'False'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = RemoveStopWordsOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        stop_words = []
        output_1 = input_1.copy()
         
        word_tokens = output_1['text2'].values       
        result = []
        for row in word_tokens:
            result.append([w for w in row if not w in stop_words])
        output_1['col3'] = result
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


"""
    NGram Operations tests
"""


def test_n_gram_operations_minimum_success():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: ['text1'],
        GenerateNGramsOperation.N_PARAM: '5'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = GenerateNGramsOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         
         grams = []
         for row in output_1['text1'].values:
             grams.append([" ".join(gram) for gram in ngrams(row, 5)])
         
         output_1['text1_ngram'] = grams           
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_n_gram_operations_success():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: ['text'],
        GenerateNGramsOperation.ALIAS_PARAM: 'c',
        GenerateNGramsOperation.N_PARAM: '10'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = GenerateNGramsOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         
         grams = []
         for row in output_1['text'].values:
             grams.append([" ".join(gram) for gram in ngrams(row, 10)])
         
         output_1['c'] = grams
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_n_gram_missing_param_failure():
    params = {
        GenerateNGramsOperation.ATTRIBUTES_PARAM: ['text'],
        GenerateNGramsOperation.ALIAS_PARAM: 'c'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        GenerateNGramsOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


"""
    WordToVector Operation Operation Tests
"""


def test_word_to_vector_missing_param_failure():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'count',
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        WordToVectorOperation(params, named_inputs=n_in, named_outputs=n_out)


def test_word_to_vector_count_minimum_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'count',
        WordToVectorOperation.ATTRIBUTES_PARAM: ['text'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'vocabulary': 'vocab_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
        
         def do_nothing(tokens):
             return tokens
         
         corpus = output_1['text'].values.tolist()
         vector_model_1 = CountVectorizer(tokenizer=do_nothing,
                          preprocessor=None, lowercase=False, 
                          min_df=1, max_features=1000)
         
         vector_model_1.fit(corpus)
         output_1['text_vec'] = vector_model_1.transform(corpus).toarray(
         ).tolist()
         vocab_1 = vector_model_1.get_feature_names()
         """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_word_to_vector_count_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: 'count',
        WordToVectorOperation.ATTRIBUTES_PARAM: ['text'],
        WordToVectorOperation.ALIAS_PARAM: 'c',
        WordToVectorOperation.VOCAB_SIZE_PARAM: 6,
        WordToVectorOperation.MINIMUM_DF_PARAM: 5,
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1', 'vocabulary': 'vocab_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
        
         def do_nothing(tokens):
             return tokens
         
         corpus = output_1['text'].values.tolist()
         vector_model_1 = CountVectorizer(tokenizer=do_nothing,
                          preprocessor=None, lowercase=False, 
                          min_df=5, max_features=6)
         
         vector_model_1.fit(corpus)
         output_1['c'] = vector_model_1.transform(corpus).toarray().tolist()
         vocab_1 = vector_model_1.get_feature_names()
         """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_word_to_vector_w2v_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: WordToVectorOperation.TYPE_WORD2VEC,
        WordToVectorOperation.ATTRIBUTES_PARAM: ['col_1'],
        WordToVectorOperation.ALIAS_PARAM: 'col_2',
        WordToVectorOperation.VOCAB_SIZE_PARAM: 10,
        WordToVectorOperation.MINIMUM_DF_PARAM: 5,

    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        output_1 = input_1.copy()
        dim = 10
        corpus = output_1['col_1'].values.tolist()
        vector_model_1 = Word2Vec(corpus, min_count=5, 
             max_vocab_size=10, size=dim)
         
        vector = [np.mean([vector_model_1.wv[w] for w in words if w in 
        vector_model_1.wv]
                   or [np.zeros(dim)], axis=0) for words in corpus]     
        output_1['col_2'] = vector
        vocab_task_1 = [w for w in vector_model_1.wv.vocab]
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_word_to_vector_tfidf_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: WordToVectorOperation.TYPE_TFIDF,
        WordToVectorOperation.ATTRIBUTES_PARAM: ['col_1'],
        WordToVectorOperation.ALIAS_PARAM: 'col_2',
        WordToVectorOperation.VOCAB_SIZE_PARAM: 200,
        WordToVectorOperation.MINIMUM_DF_PARAM: 5,

    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
         
         def do_nothing(tokens):
             return tokens
         
         corpus = output_1['col_1'].values.tolist()
         vector_model_1 = TfidfVectorizer(tokenizer=do_nothing,
                          preprocessor=None, lowercase=False, 
                          min_df=5, max_features=200)
         
         vector_model_1.fit(corpus)
         output_1['col_2'] = vector_model_1.transform(corpus).toarray().tolist()
         vocab_task_1 = vector_model_1.get_feature_names()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_word_to_vector_hashing_operation_success():
    params = {
        WordToVectorOperation.TYPE_PARAM: WordToVectorOperation.TYPE_HASHING_TF,
        WordToVectorOperation.ATTRIBUTES_PARAM: ['col_1'],
        WordToVectorOperation.ALIAS_PARAM: 'col_3',
        WordToVectorOperation.VOCAB_SIZE_PARAM: 6,
        WordToVectorOperation.MINIMUM_DF_PARAM: 5,
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = WordToVectorOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
         output_1 = input_1.copy()
        
         def do_nothing(tokens):
             return tokens
         
         corpus = output_1['col_1'].values.tolist()
         vector_model_1 = HashingVectorizer(tokenizer=do_nothing,
                          preprocessor=None, lowercase=False, 
                          n_features=6)
         
         vector_model_1.fit(corpus)
         output_1 = input_1.copy()
         vector = vector_model_1.transform(corpus).toarray().tolist()
         output_1['col_3'] = vector
         
         # There is no vocabulary in this type of transformer
         vocab_task_1 = None
         """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)