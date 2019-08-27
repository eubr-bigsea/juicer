# coding=utf-8


import json

try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import zip_longest as zip_longest
from textwrap import dedent

from juicer.operation import Operation


class TokenizerOperation(Operation):
    """
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

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_SIMPLE)
        if self.type not in [self.TYPE_REGEX, self.TYPE_SIMPLE]:
            raise ValueError(
                _('Invalid type for operation Tokenizer: {}').format(self.type))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with attribute name suffixed by tokenized.
        self.alias = [x[1] or '{}_tokenized'.format(x[0]) for x in
                      zip_longest(self.attributes,
                                  self.alias[:len(self.attributes)])]

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, r'\s+')
        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        """
        Generate code with RegexTokenizer because TextTokenizer does not have
        the parameter minTokenLength.
        """
        input_data = self.named_inputs['input data']

        if self.type == self.TYPE_SIMPLE:
            self.expression_param = r'\s+'
        code = """
                col_alias = {alias}
                pattern_exp = r'{pattern}'
                min_token_length = {min_token}
                tokenizers = [RegexTokenizer(inputCol=col, outputCol=alias,
                                    pattern=pattern_exp,
                                    minTokenLength=min_token_length)
                                    for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=tokenizers)

                {out} = pipeline.fit({input}).transform({input})
            """.format(input=input_data, out=self.output,
                       alias=json.dumps(list(zip(self.attributes, self.alias))),
                       pattern=self.expression_param,
                       min_token=self.min_token_lenght)

        return dedent(code)


class RemoveStopWordsOperation(Operation):
    """
    Stop words are words which should be excluded from the input,
    typically because the words appear frequently and don’t carry
    as much meaning.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'
    STOP_WORD_LIST_PARAM = 'stop_word_list'
    STOP_WORD_ATTRIBUTE_PARAM = 'stop_word_attribute'
    STOP_WORD_LANGUAGE_PARAM = 'language'
    STOP_WORD_CASE_SENSITIVE_PARAM = 'sw_case_sensitive'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.stop_word_attribute = self.parameters.get(
            self.STOP_WORD_ATTRIBUTE_PARAM, 'stop_word')

        self.stop_word_list = [s.strip() for s in
                               self.parameters.get(self.STOP_WORD_LIST_PARAM,
                                                   '').split(',') if s.strip()]

        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by
        # _no_stopwords.
        self.alias = [x[1] or '{}_no_stopwords'.format(x[0]) for x in
                      zip_longest(self.attributes,
                                  self.alias[:len(self.attributes)])]

        self.stop_word_language = self.parameters.get(
            self.STOP_WORD_LANGUAGE_PARAM, 'english')

        self.sw_case_sensitive = self.parameters.get(
            self.STOP_WORD_CASE_SENSITIVE_PARAM, 'False')

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = ''

        if len(self.named_inputs) != 2:
            """
            Loads the default stop words for the given language.
            Supported languages: danish, dutch, english, finnish,
            french, german, hungarian, italian, norwegian, portuguese,
            russian, spanish, swedish, turkish
            """
            if self.stop_word_list:
                code = "sw = {list}".format(
                    list=json.dumps(self.stop_word_list))
            elif self.stop_word_language:
                code = dedent("""
                sw = StopWordsRemover.loadDefaultStopWords('{}')""".format(
                    self.stop_word_language))

            code += dedent("""
            col_alias = {alias}
            case_sensitive = {case_sensitive}
            removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                        stopWords=sw,
                        caseSensitive=case_sensitive)
                        for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=removers)
            {out} = pipeline.fit({input}).transform({input})
        """.format(input=input_data,
                   out=self.output,
                   alias=json.dumps(list(zip(self.attributes, self.alias))),
                   case_sensitive=self.sw_case_sensitive))
        else:
            code = ("sw = [stop[0].strip() "
                    "for stop in {}.collect() if stop and stop[0]]").format(
                self.named_inputs['stop words'])

            code += dedent("""
                col_alias = {3}
                case_sensitive = {4}
                removers = [StopWordsRemover(inputCol=col, outputCol=alias,
                                stopWords=sw, caseSensitive=case_sensitive)
                                for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=removers)
                {2} = pipeline.fit({1}).transform({1})
            """.format(self.attributes, input_data,
                       self.output,
                       json.dumps(list(zip(self.attributes, self.alias))),
                       self.sw_case_sensitive))
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
                      zip_longest(self.attributes,
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
        if self.type == self.TYPE_COUNT:
            code = dedent("""
                col_alias = {alias}
                vectorizers = [
                    CountVectorizer(minTF={min_tf}, minDF={min_df},
                        vocabSize={vocab_size}, binary=False, inputCol=col,
                        outputCol=alias) for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=vectorizers)
                {model} = pipeline.fit({input})
                {out} = {model}.transform({input})
                {vocab} = dict([(col_alias[i][1], v.vocabulary)
                        for i, v in enumerate({model}.stages)])""".format(
                input=input_data,
                out=self.output,
                alias=json.dumps(list(zip(self.attributes, self.alias))),
                min_tf=self.minimum_tf, min_df=self.minimum_df,
                vocab_size=self.vocab_size,
                model=self.output_model,
                vocab=self.vocab))
        elif self.type == self.TYPE_HASHING_TF:
            code = dedent("""
                col_alias = {alias}
                hashing_transformers = [
                    HashingTF(numFeatures={vocab_size}, binary=False,
                    inputCol=col, outputCol=alias) for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=hashing_transformers)
                {model} = pipeline.fit({input})
                {out} = {model}.transform({input})

                # There is no vocabulary in this type of transformer
                {vocab} = {{}}""".format(
                input=input_data,
                out=self.output,
                alias=json.dumps(list(zip(self.attributes, self.alias))),
                vocab_size=self.vocab_size,
                model=self.output_model,
                vocab=self.vocab))
        elif self.type == self.TYPE_WORD2VEC:
            # @FIXME Check
            code = dedent("""
                col_alias = {aliases}
                vectorizers = [Word2Vec(vectorSize={size},
                            minCount={count},
                            numPartitions=1,
                            stepSize=0.025,
                            maxIter=1,
                            seed=None,
                            inputCol=col,
                            outputCol=alias) for col, alias in col_alias]
                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=vectorizers)
                {model} = pipeline.fit({input})
                {out} = {model}.transform({input})
                {vocab} = dict([(col_alias[i][1], v.getVectors())
                             for i, v in enumerate({model}.stages)])""".format(
                self.attributes,
                input=input_data,
                out=self.output,
                aliases=json.dumps(list(zip(self.attributes, self.alias))),
                size=self.minimum_size, count=self.minimum_count,
                vocab=self.vocab,
                model=self.output_model
            ))

        else:
            raise ValueError(
                _("Invalid type '{}' for task {}").format(self.type,
                                                          self.__class__))
        return code


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
        if self.N_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.N_PARAM, self.__class__))

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.n = int(self.parameters.get(self.N_PARAM, 2))
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _ngram.
        self.alias = [x[1] or '{}_ngram'.format(x[0]) for x in
                      zip_longest(self.attributes,
                                  self.alias[:len(self.attributes)])]

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = dedent("""
            col_alias = {alias}
            n_gramers = [NGram(n={n}, inputCol=col,
                           outputCol=alias) for col, alias in col_alias]
            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=n_gramers)
            model = pipeline.fit({input})
            {output} = model.transform({input})
            """).format(
            alias=json.dumps(list(zip(self.attributes, self.alias))),
            n=self.n, input=input_data, output=self.output)

        return code
