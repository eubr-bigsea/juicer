# -*- coding: utf-8 -*-
import ast
import pprint
from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


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

        self.has_code = 'input data' in self.named_inputs \
                        and self.contains_results()
        if self.has_code:

            self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
            if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
                raise ValueError(_('Invalid type for '
                                   'operation Tokenizer: {}').format(self.type))

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\s+')

            self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format(self.ATTRIBUTES_PARAM, self.__class__))

            self.alias = [alias.strip() for alias in
                          parameters.get(self.ALIAS_PARAM, '').split(',')]
            # Adjust alias in order to have the same number of aliases as
            # attributes by filling missing alias with the attribute name
            # sufixed by _indexed.
            self.alias = [x[1] or '{}_tok'.format(x[0]) for x in
                          izip_longest(self.attributes,
                                       self.alias[:len(self.attributes)])]

            self.expression_param = parameters.get(self.EXPRESSION_PARAM, '\s+')
            if len(self.expression_param) == 0:
                self.expression_param = '\s+'

            self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        """Generate code."""

        if self.min_token_lenght is not None:
            self.min_token_lenght = \
                ' if len(word) > {}'.format(self.min_token_lenght)

        if self.type == self.TYPE_SIMPLE:
            code = """
            {output} = {input}.copy()
            result = []
            from nltk.tokenize import ToktokTokenizer
            toktok = ToktokTokenizer()
    
            for row in {output}['{att}'].values:
                result.append([word 
                for word in toktok.tokenize(row){limit}])
            {output}['{alias}'] = result
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       att=self.attributes[0], alias=self.alias[0],
                       limit=self.min_token_lenght)
        else:
            code = """
           {output} = {input}.copy()
           result = []
           from nltk.tokenize import regexp_tokenize

           for row in {output}['{att}'].values:
               result.append([word for word in 
                              regexp_tokenize(row, pattern='{exp}'){limit}])

           {output}['{alias}'] = result
           """.format(output=self.output,
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
    STOP_WORD_CASE_SENSITIVE_PARAM = 'sw_case_sensitive'
    LANG_PARAM = 'language'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = 'input data' in self.named_inputs \
                        and self.contains_results()

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

    def generate_code(self):
        """Generate code."""
        code = """
        stop_words = []
        {OUT} = {IN}.copy()
        """.format(OUT=self.output, IN=self.named_inputs['input data'])
        if len(self.lang) > 0:
            code += """
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words += stopwords.words('{language}')""".format(
                    language=self.lang.lower())

        if len(self.named_inputs) == 2 and len(self.stop_word_attribute) > 0:
            code += """
        stop_words += {in2}['{att2}'].values.tolist()
        print (stop_words)
        """.format(in2=self.stopwords_input, att2=self.stop_word_attribute)
        if len(self.stop_word_list) > 1:
            code += """
        stop_words += {stop_word_list}
        """.format(stop_word_list=self.stop_word_list)

        if self.sw_case_sensitive:
            code += """
        word_tokens = {OUT}['{att}'].values       
        result = []
        for row in word_tokens:
            result.append([w for w in row if not w in stop_words])
        {OUT}['{alias}'] = result
        """.format(att=self.attributes,
                   att_stop=self.stop_word_attribute,
                   alias=self.alias, case=self.sw_case_sensitive,
                   OUT=self.output,
                   stoplist=self.stop_word_list, sw=self.stopwords_input)
        else:
            code += """
        stop_words = [w.lower() for w in stop_words]
        word_tokens = {OUT}['{att}'].values       
        result = []
        for row in word_tokens:
            result.append([w for w in row if not w in stop_words])
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

        self.has_code = 'input data' in self.named_inputs \
                        and self.contains_results()
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

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
            {output} = {input}.copy()
            from nltk.util import ngrams
            
            grams = []
            for row in {output}['{att}'].values:
                grams.append(list(ngrams(row, {n})))
                
            {output}['{alias}'] = grams
            """).format(att=self.attributes, alias=self.alias,
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
    MINIMUM_TF_PARAM = 'minimum_tf'

    MINIMUM_VECTOR_SIZE_PARAM = 'minimum_size'
    MINIMUM_COUNT_PARAM = 'minimum_count'

    TYPE_COUNT = 'count'
    TYPE_WORD2VEC = 'word2vec'
    TYPE_HASHING_TF = 'hashing_tf'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_COUNT)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _vect.
        self.alias = [x[1] or '{}_vect'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000) or 1000
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 1) or 1.0
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1) or 1.0

        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 100)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 5)

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))
        self.vocab = self.named_outputs.get('vocabulary',
                                            'vocab_task_{}'.format(self.order))
        self.output_model = self.named_outputs.get(
            'vector-model', 'vector_model_{}'.format(self.order))

        if self.type not in [self.TYPE_HASHING_TF, self.TYPE_WORD2VEC,
                             self.TYPE_COUNT]:
            raise ValueError(
                _("Invalid type '{}' for task {}").format(self.type,
                                                          self.__class__))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.vocab, self.output_model])

    def generate_code(self):
        input_data = self.named_inputs['input data']

        # max_df is used for removing terms that appear too frequently
        if self.type == self.TYPE_COUNT:
            code = dedent("""
            from sklearn.feature_extraction.text import CountVectorizer
            corpus = {input}['{att}'].apply(lambda x: ' '.join(x))
            bow_transformer = CountVectorizer(min_df={min_df}, 
            binary=False, max_features={vocab_size})
                            
            bow_transformer.fit(corpus)
            {out} = {input}.copy()
            vector = bow_transformer.transform(corpus).toarray().tolist()
            {out}['{alias}'] = vector
            {model} = bow_transformer
            {vocab} = bow_transformer.get_feature_names()
            """.format(
                    input=input_data,
                    out=self.output,
                    att=self.attributes[0],
                    alias=self.alias[0],
                    min_df=self.minimum_df,
                    vocab_size=self.vocab_size,
                    model=self.output_model,
                    vocab=self.vocab))

        elif self.type == self.TYPE_HASHING_TF:
            code = dedent("""
            from sklearn.feature_extraction.text import HashingVectorizer
            corpus = {input}['{att}'].apply(lambda x: ' '.join(x))
            model = HashingVectorizer(binary=False, n_features={vocab_size})
                
            model.fit(corpus)
            {out} = {input}.copy()
            vector = model.transform(corpus).toarray().tolist()
            {out}['{alias}'] = vector
            {model} = model

            # There is no vocabulary in this type of transformer
            {vocab} = {{}}
            """.format(
                    input=input_data,
                    out=self.output,
                    att=self.attributes[0],
                    alias=self.alias[0],
                    vocab_size=self.vocab_size,
                    model=self.output_model,
                    vocab=self.vocab))
        elif self.type == self.TYPE_WORD2VEC:
            # @FIXME Check
            code = dedent("""
                """.format(
                self.attributes,
                input=input_data,
                out=self.output,
                aliases=json.dumps(zip(self.attributes, self.alias)),
                size=self.minimum_size, count=self.minimum_count,
                vocab=self.vocab,
                model=self.output_model
            ))

        else:
            raise ValueError(
                _("Invalid type '{}' for task {}").format(self.type,
                                                          self.__class__))
        return code
