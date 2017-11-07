# coding=utf-8
import json
from itertools import izip_longest
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
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

        self.expression_param = parameters.get(self.EXPRESSION_PARAM, r'\s+')
        self.min_token_lenght = parameters.get(self.MINIMUM_SIZE, 3)
        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = ''

        if self.type == self.TYPE_SIMPLE:

            code = """
                col_alias = {3}
                tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                                    for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=tokenizers)

                {2} = pipeline.fit({1}).transform({1})
            """.format(self.attributes, input_data, self.output,
                       json.dumps(zip(self.attributes, self.alias)))

        elif self.type == self.TYPE_REGEX:
            code = """
                col_alias = {3}
                pattern_exp = r'{4}'
                min_token_length = {5}
                regextokenizers = [RegexTokenizer(inputCol=col, outputCol=alias,
                                    pattern=pattern_exp,
                                    minTokenLength=min_token_length)
                                    for col, alias in col_alias]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=regextokenizers)

                {2} = pipeline.fit({1}).transform({1})
            """.format(self.attributes, input_data, self.output,
                       json.dumps(zip(self.attributes, self.alias)),
                       self.expression_param,
                       self.min_token_lenght)

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
                      izip_longest(self.attributes,
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
                   alias=json.dumps(zip(self.attributes, self.alias)),
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
                       json.dumps(zip(self.attributes, self.alias)),
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

        self.vocab_size = parameters.get(self.VOCAB_SIZE_PARAM, 1000)
        self.minimum_df = parameters.get(self.MINIMUM_DF_PARAM, 5)
        self.minimum_tf = parameters.get(self.MINIMUM_TF_PARAM, 1)

        self.minimum_size = parameters.get(self.MINIMUM_VECTOR_SIZE_PARAM, 3)
        self.minimum_count = parameters.get(self.MINIMUM_COUNT_PARAM, 0)

        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))
        self.vocab = self.named_outputs.get('vocabulary',
                                            'vocab_task_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return sep.join([self.output, self.vocab])

    def generate_code(self):
        input_data = self.named_inputs['input data']

        if self.type == self.TYPE_COUNT:
            code = dedent("""
                col_alias = {3}
                vectorizers = [CountVectorizer(minTF={4}, minDF={5},
                               vocabSize={6}, binary=False, inputCol=col,
                               outputCol=alias) for col, alias in col_alias]""")

            code += dedent("""
            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=vectorizers)
            model = pipeline.fit({1})
            {2} = model.transform({1})
            """)
            code += dedent("""
                {} = dict([(col_alias[i][1], v.vocabulary)
                        for i, v in enumerate(model.stages)])""".format(
                self.vocab))

            code = code.format(self.attributes, input_data,
                               self.output,
                               json.dumps(zip(self.attributes, self.alias)),
                               self.minimum_tf, self.minimum_df,
                               self.vocab_size)

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
                model = pipeline.fit({input})
                {out} = model.transform({input})
                {vocab} = dict([(col_alias[i][1], v.getVectors())
                             for i, v in enumerate(model.stages)])""".format(
                self.attributes,
                input=input_data,
                out=self.output,
                aliases=json.dumps(zip(self.attributes, self.alias)),
                size=self.minimum_size, count=self.minimum_count,
                vocab=self.vocab
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
                      izip_longest(self.attributes,
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
            """).format(alias=json.dumps(zip(self.attributes, self.alias)),
                        n=self.n, input=input_data, output=self.output)

        return code
