# -*- coding: utf-8 -*-

import ast
import pprint
import re
import numpy as np
from textwrap import dedent
from juicer.operation import Operation
from itertools import zip_longest


class WordCountingOperation(Operation):
    """ Counts the words
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))


        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 
        self.transpiler_utils.add_import("from collections import Counter")
        self.transpiler_utils.add_import("from nltk.tokenize import RegexpTokenizer")
        self.transpiler_utils.add_import("import nltk")


    def generate_code(self):
        """Generate code."""

        code = f"""
        tokenizer = RegexpTokenizer(r'\w+', flags=re.IGNORECASE)

        df = {self.input}
        for attr in {self.attributes}:
            df[attr + '_word_count'] = df[attr].astype(str).str.findall(r'(\w+)').str.len()


        {self.output} = df
        """
        return dedent(code)


class LowerCaseOperation(Operation):
    """ Lowers the case of words
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_lowercase'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""

        code = f"""

        df = {self.input}
        for attr in {self.attributes}:
            df[attr] = df[attr].astype(str).str.lower()
	
        {self.output} = df
        """
        return dedent(code)


class TokenizeOperation(Operation):
    """TokenizerOperation.
    The tokenization operation breaks a paragraph of text into smaller pieces, such as words or sentences. Token is a single entity that is
    building blocks for sentence or paragraph.This operation classifying tokens into their parts of speech.
    A simple Tokenizer class provides this functionality.
    """

    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_tokenize'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        from nltk.tokenize import TweetTokenizer
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize

        def tokenizer(word):
            tokens=nltk.word_tokenize(word)
            
            return tokens
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(tokenizer)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class SynonymsOperation(Operation):
    """This operation generate synonyms of a word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_synonym'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        import nltk
        nltk.download('wordnet')
        from nltk.corpus import wordnet
        def synonym(word):
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
    
            return synonyms

        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(synonym)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class AntonymsOperation(Operation):
    """This operation generate antonyms of a word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)


        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')]

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_antonyms'.format(x[0]) for x
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])]

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        """Generate code."""
        code = """
        import nltk
        nltk.download('wordnet')
        from nltk.corpus import wordnet
        def antonyms(word):
            antonyms = []
            synonyms = []
            aux = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                     synonyms.append(l.name())
                     if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            aux = antonyms
            return aux
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(antonyms)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class DefinerOperation(Operation):

    """This operation generate synonyms of a word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_definition'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        import nltk
        nltk.download('wordnet')
        nltk.download('omw')
        from nltk.corpus import wordnet
        

        def definer(word):
            syn_arr = wordnet.synsets(word)
           
            
            return syn_arr[0].definition()
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(definer)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class NerOperation(Operation):

    """This operation generate synonyms of a word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_ner'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """

        import spacy
        import spacy.cli
        spacy.cli.download("pt_core_news_sm")
        nlp = spacy.load('pt_core_news_sm')

        def ner(word):
            r = []
            documento = nlp(word)
            for named_entity in documento.ents:
    
                r.append(named_entity)
                r.append(named_entity.label_)
            
            return r


        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(ner)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)


class StemmingOperation(Operation):
    """Stemming is the process of producing morphological variants of a root/base word.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
       
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _stems.
        self.alias = [x[1] or '{}_stems'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 
    def generate_code(self):
        """Generate code."""

        code = """
        stemmer = SnowballStemmer("english")
        def stemming(word):
            return stemmer.stem(word)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(stemming)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class LemmatizationOperation(Operation):
    """Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_lemmas'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        def lemmatization(word):
            return lemmatizer.lemmatize(word)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(lemmatization)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)


class NormalizationOperation(Operation):
    """Normalization is the process of removing punctuaction and lower casing tokens.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_normalized'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        def normalization(sentence):
            puncts='.?!'
            for punct in puncts:
                sentence = sentence.replace(punct, ' ')
            return sentence.lower()

        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(normalization)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class PosTaggingOperation(Operation):
    """Part-of-speech (POS) tagging is a process of labeling each word in a sentence with a morphosyntactic class
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_posTags'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        nltk.download('averaged_perceptron_tagger')
        def posTagging(sentence):
            aux = eval(str(sentence))
            return nltk.pos_tag(aux)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(posTagging)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)

class WordSegmentationOperation(Operation):
    """Word Segmentation is the process of dividing a string into its word components.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, []) or []
        if not self.attributes:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}")
                .format(self.ATTRIBUTES_PARAM, self.__class__))

        self.alias = [ alias.strip() for alias in parameters.get(self.ALIAS_PARAM, '').split(',')] 

        # Adjust alias in order to have the same number of aliases as attributes 
        # by filling missing alias with the attribute name suffixed by _pdf.
        self.alias = [x[1] or '{}_segmentedwords'.format(x[0]) for x 
                in zip_longest(self.attributes, self.alias[:len(self.attributes)])] 

        self.output = self.named_outputs.get(
                'output data', 'output_data_{}'.format(self.order))

        self.input = self.named_inputs.get(
                'input data', 'input_data_{}'.format(self.order))
        self.has_code = any([len(named_outputs) > 0, self.contains_results()]) 

    def generate_code(self):
        """Generate code."""
        code = """
        load()
        def segmentWord(sentence):
            return segment(sentence)
        {out} = {input}
        alias = {alias}
        for i, attr in enumerate({attributes}):
            {out}[alias[i]] = {input}[attr].apply(segmentWord)
        """.format(attributes=self.attributes,
                alias = self.alias,
                input = self.input,
                out=self.output)
        return dedent(code)
