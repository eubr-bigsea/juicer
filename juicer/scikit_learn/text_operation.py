# -*- coding: utf-8 -*-
import ast
import pprint
import numpy as np
from textwrap import dedent
from juicer.operation import Operation
from itertools import zip_longest


class TokenizerOperation(Operation):
    """TokenizerOperation.

    Tokenization is the process of taking text (such as a sentence) and
    breaking it into individual terms (usually words). A simple Tokenizer
    class provides this functionality.
    """

    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'
    MINIMUM_SIZE = 'min_token_length'

    TYPE_SIMPLE = 'simple'
    TYPE_REGEX = 'regex'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        if self.has_code:

            self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
            if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
                raise ValueError(_('Invalid type for '
                                   'operation Tokenizer: {}').format(self.type))

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\\s+')

            self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                            self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = [alias.strip() for alias in
                          parameters.get(self.ALIAS_PARAM, '').split(',')]
            # Adjust alias in order to have the same number of aliases as
            # attributes by filling missing alias with the attribute name
            # sufixed by _indexed.
            self.alias = [x[1] or '{}_tok'.format(x[0]) for x in
                          zip_longest(self.attributes,
                                      self.alias[:len(self.attributes)])]

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\\s+')
            if len(self.expression_param) == 0:
                self.expression_param = '\\s+'

            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            if self.min_token_lenght is not None:
                self.min_token_lenght = \
                    ' if len(word) >= {}'.format(self.min_token_lenght)

            if self.type == self.TYPE_SIMPLE:
                code = """
                from nltk.tokenize import TweetTokenizer
                {output} = {input}{copy_code}
                result = []
                toktok = TweetTokenizer()
        
                for row in {output}['{att}'].to_numpy():
                    result.append([word 
                    for word in toktok.tokenize(row){limit}])
                {output}['{alias}'] = result
                """.format(copy_code=copy_code, output=self.output,
                           input=self.named_inputs['input data'],
                           att=self.attributes[0], alias=self.alias[0],
                           limit=self.min_token_lenght)
            else:
                code = """
                from nltk.tokenize import regexp_tokenize
                {output} = {input}{copy_code}
                result = []
    
                for row in {output}['{att}'].to_numpy():
                    result.append([word for word in 
                                  regexp_tokenize(row, pattern=r'{exp}'){limit}])
    
                {output}['{alias}'] = result
                """.format(copy_code=copy_code, output=self.output,
                          input=self.named_inputs['input data'],
                          att=self.attributes[0], alias=self.alias[0],
                          exp=self.expression_param, limit=self.min_token_lenght)

            return dedent(code)


class RemoveStopWordsOperation(Operation):
    """RemoveStopWordsOperation.

    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don't carry
    as much meaning.

    """

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    STOP_WORD_LIST_PARAM = 'stop_word_list'
    STOP_WORD_ATTRIBUTE_PARAM = 'stop_word_attribute'
    STOP_WORD_LANGUAGE_PARAM = 'stop_word_language'
    STOP_WORD_CASE_SENSITIVE_PARAM = 'case-sensitive'
    LANG_PARAM = 'language'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            self.output = self.named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)[0]
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(self.ATTRIBUTES_PARAM, self.__class__))

            self.sw_case_sensitive = self.parameters.get(
                    self.STOP_WORD_CASE_SENSITIVE_PARAM, False)

            self.stop_word_list = [s.strip() for s in
                                   self.parameters.get(
                                           self.STOP_WORD_LIST_PARAM,
                                           '').split(',')]

            self.alias = parameters.get(self.ALIAS_PARAM, 'tokenized_rm')
            self.stopwords_input = self.named_inputs.get('stop words', None)
            self.stop_word_attribute = self.parameters.get(
                    self.STOP_WORD_ATTRIBUTE_PARAM, [''])[0]
            self.lang = self.parameters.get(self.LANG_PARAM, '') or ''

            self.transpiler_utils.add_import(
                    "import nltk\nnltk.download('stopwords')")
            self.transpiler_utils.add_import(
                    "from nltk.corpus import stopwords")

    def generate_code(self):
        """Generate code."""
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """
            stop_words = []
            {OUT} = {IN}{copy_code}
            """.format(copy_code=copy_code, OUT=self.output,
                       IN=self.named_inputs['input data'])
            if len(self.lang) > 0:
                code += """
            stop_words += stopwords.words('{language}')""".format(
                        language=self.lang.lower())

            if len(self.named_inputs) == 2 and len(self.stop_word_attribute) > 0:
                code += """
            stop_words += {in2}['{att2}'].to_numpy().tolist()
            """.format(in2=self.stopwords_input, att2=self.stop_word_attribute)
            if len(self.stop_word_list) > 1:
                code += """
            stop_words += {stop_word_list}
            """.format(stop_word_list=self.stop_word_list)

            if self.sw_case_sensitive == "1":
                code += """
            word_tokens = {OUT}['{att}'].to_numpy()       
            result = []
            case_sensitive = True
            for row in word_tokens:
                itr = []
                for w in row:
                    if not w in stop_words:
                        itr.append(w)
                result.append(itr)
            {OUT}['{alias}'] = result
            """.format(att=self.attributes,
                       att_stop=self.stop_word_attribute,
                       alias=self.alias, case=self.sw_case_sensitive,
                       OUT=self.output,
                       stoplist=self.stop_word_list, sw=self.stopwords_input)
            else:
                code += """
            stop_words = [w.lower() for w in stop_words]
            word_tokens = {OUT}['{att}'].to_numpy()       
            result = []
            case_sensitive = False
            for row in word_tokens:
                itr = []
                for w in row:
                    if not w.lower() in stop_words:
                        itr.append(w)
                result.append(itr)                
            {OUT}['{alias}'] = result
            """.format(att=self.attributes,
                       att_stop=self.stop_word_attribute,
                       alias=self.alias, case=self.sw_case_sensitive,
                       OUT=self.output,
                       stoplist=self.stop_word_list, sw=self.stopwords_input)

            return dedent(code)


class GenerateNGramsOperation(Operation):
    """ Generates N-Grams from word vectors
    An n-gram is a sequence of n tokens (typically words) for some
    integer n. The NGram class can be used to transform input features
    into n-grams.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    N_PARAM = 'n'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            self.output = self.named_outputs.get('output data',
                                                 'out_{}'.format(self.order))
            if self.N_PARAM in parameters:
                self.n = abs(int(self.parameters.get(self.N_PARAM, 2)))
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.N_PARAM, self.__class__))

            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)[0]
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = parameters.get(
                        self.ALIAS_PARAM, '{}_ngram'.format(self.attributes))

            self.transpiler_utils.add_import("from nltk.util import ngrams")

    def generate_code(self):
        if self.has_code:
            input_data = self.named_inputs['input data']
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = dedent("""
                from nltk.util import ngrams
                {output} = {input}{copy_code}
                        
                grams = []
                for row in {output}['{att}'].to_numpy():
                    grams.append([" ".join(gram) for gram in ngrams(row, {n})])
                    
                {output}['{alias}'] = grams
                """).format(copy_code=copy_code,
                            att=self.attributes, alias=self.alias,
                            n=self.n, input=input_data, output=self.output)

            return code


class WordToVectorOperation(Operation):
    """
    Word2Vec is an Estimator which takes sequences of words that
    represents documents and trains a Word2VecModel. The model is
    a Map(String, Vector) essentially, which maps each word to an
    unique fix-sized vector. The Word2VecModel transforms each
    documents into a vector using the average of all words in the
    document, which aims to other computations of documents such
    as similarity calculation consequencely.
    """
    TYPE_PARAM = 'type'
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    VOCAB_SIZE_PARAM = 'vocab_size'
    MINIMUM_DF_PARAM = 'minimum_df'

    MINIMUM_VECTOR_SIZE_PARAM = 'minimum_size'

    TYPE_COUNT = 'count'
    TYPE_TFIDF = 'TF-IDF'
    TYPE_WORD2VEC = 'word2vec'
    TYPE_HASHING_TF = 'hashing_tf'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = 'input data' in self.named_inputs \
                        and any([self.contains_results(),
                                 'output data' in self.named_outputs])
        if self.has_code:
            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)[0]
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.ATTRIBUTES_PARAM, self.__class__))

            self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)

            self.alias = parameters.get(
                    self.ALIAS_PARAM, '{}_vec'.format(self.attributes))

            self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM,
                                             1000) or 1000
            self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM,
                                             1) or 1.0

            self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM,
                                               100)

            self.output = self.named_outputs.get('output data',
                                                 'out_{}'.format(self.order))
            self.vocab = self.named_outputs.get('vocabulary',
                                                'vocab_task_{}'.format(self.order))
            self.output_model = self.named_outputs.get(
                'vector-model', 'vector_model_{}'.format(self.order))

            if self.type not in [self.TYPE_HASHING_TF, self.TYPE_WORD2VEC,
                                 self.TYPE_COUNT, self.TYPE_TFIDF]:
                raise ValueError(
                    _("Invalid type '{}' for task {}").format(self.type,
                                                              self.__class__))

            if self.type == self.TYPE_COUNT:
                self.transpiler_utils.add_import(
                    'from sklearn.feature_extraction.text '
                    'import CountVectorizer')
            elif self.type == self.TYPE_HASHING_TF:
                self.transpiler_utils.add_import(
                    'from sklearn.feature_extraction.text '
                    'import HashingVectorizer')
            elif self.type == self.TYPE_WORD2VEC:
                self.transpiler_utils.add_custom_function(
                        'get_w2v_vector', get_w2v_vector)
                self.transpiler_utils.add_import(
                    'from gensim.models import Word2Vec')
            elif self.type == self.TYPE_TFIDF:
                self.transpiler_utils.add_import(
                    'from sklearn.feature_extraction.text '
                    'import TfidfVectorizer')

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.vocab, self.output_model])

    def generate_code(self):
        if self.has_code:
            input_data = self.named_inputs['input data']
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            if self.type == self.TYPE_COUNT:
                code = dedent("""
                {out} = {input}{copy_code}
                
                def do_nothing(tokens):
                    return tokens
                
                corpus = {out}['{att}'].to_numpy().tolist()
                {model} = CountVectorizer(tokenizer=do_nothing,
                                 preprocessor=None, lowercase=False, 
                                 min_df={min_df}, max_features={vocab_size})
                                
                {model}.fit(corpus)
                {out}['{alias}'] = {model}.transform(corpus).toarray().tolist()
                {vocab} = {model}.get_feature_names()
                """.format(
                        copy_code=copy_code,
                        input=input_data,
                        out=self.output,
                        att=self.attributes,
                        alias=self.alias,
                        min_df=self.minimum_df,
                        vocab_size=self.vocab_size,
                        model=self.output_model,
                        vocab=self.vocab))

            elif self.type == self.TYPE_TFIDF:
                code = dedent("""
                {out} = {input}{copy_code}
                
                def do_nothing(tokens):
                    return tokens
                
                corpus = {out}['{att}'].to_numpy().tolist()
                {model} = TfidfVectorizer(tokenizer=do_nothing,
                                 preprocessor=None, lowercase=False, 
                                 min_df={min_df}, max_features={vocab_size})
                                
                {model}.fit(corpus)
                {out}['{alias}'] = {model}.transform(corpus).toarray().tolist()
                {vocab} = {model}.get_feature_names()
                """.format(
                        copy_code=copy_code,
                        input=input_data,
                        out=self.output,
                        att=self.attributes,
                        alias=self.alias,
                        min_df=self.minimum_df,
                        vocab_size=self.vocab_size,
                        model=self.output_model,
                        vocab=self.vocab))

            elif self.type == self.TYPE_HASHING_TF:
                code = dedent("""
                
                def do_nothing(tokens):
                    return tokens
                    
                {out} = {input}{copy_code}  
                corpus = {out}['{att}'].to_numpy().tolist()
                {model} = HashingVectorizer(tokenizer=do_nothing,
                                 preprocessor=None, lowercase=False, 
                                 n_features={vocab_size})
                {model}.fit(corpus)
    
                vector = {model}.transform(corpus).toarray().tolist()
                {out}['{alias}'] = vector
    
                # There is no vocabulary in this type of transformer
                {vocab} = None
                """.format(
                        copy_code=copy_code,
                        input=input_data,
                        out=self.output,
                        att=self.attributes,
                        alias=self.alias,
                        vocab_size=self.vocab_size,
                        model=self.output_model,
                        vocab=self.vocab))
            elif self.type == self.TYPE_WORD2VEC:
                # in word2vec, the number of features its not always equals to
                # the number of vocabulary. Currently, the generated code force
                # to be equals.
                code = dedent("""
                {out} = {input}{copy_code}
                dim = {max_dim}
                corpus = {out}['{att}'].to_numpy().tolist()

                {model} = Word2Vec(corpus, min_count={min_df}, 
                    max_vocab_size={max_vocab}, size=dim)
                                
                {out}['{alias}'] = get_w2v_vector(corpus, {model}, dim)
                {vocab} = [w for w in {model}.wv.vocab]
                    """.format(copy_code=copy_code,
                               att=self.attributes,
                               input=input_data,
                               min_df=self.minimum_df,
                               alias=self.alias,
                               max_dim=self.vocab_size,
                               out=self.output,
                               max_vocab=self.vocab_size,
                               vocab=self.vocab,
                               model=self.output_model))

            else:
                raise ValueError(
                    _("Invalid type '{}' for task {}").format(self.type,
                                                              self.__class__))
            return code


def get_w2v_vector(corpus, model, dim):
    vector = [np.mean(
            [model.wv[w] for w in words if w in model.wv] or
            [np.zeros(dim)], axis=0) for words in corpus]
    return vector
