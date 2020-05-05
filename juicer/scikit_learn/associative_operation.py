# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class FrequentItemSetOperation(Operation):
    """FP-growth"""

    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'
    CONFIDENCE_PARAM = 'min_confidence'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:

            if self.MIN_SUPPORT_PARAM not in parameters:
                raise ValueError(
                        _("Parameter '{}' must be informed for task {}").format(
                                self.MIN_SUPPORT_PARAM, self.__class__))

            self.column = parameters.get(self.ATTRIBUTE_PARAM, [''])[0]
            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.9))
            self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
            if self.min_support < .0001 or self.min_support > 1.0:
                raise ValueError('Support must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.output = named_outputs.get('output data',
                                            'output_data_{}'.format(
                                                         self.order))
            self.rules_output = named_outputs.get('rules output',
                                                  'rules_{}'.format(
                                                           self.order))

            self.has_import = "import pyfpgrowth\n"

    def get_output_names(self, sep=', '):
        return sep.join([self.output,
                         self.rules_output])

    def generate_code(self):
        """Generate code."""

        if not len(self.column) > 1:
            self.column = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.column = "'{}'".format(self.column)

        code = """
        col = {col}
        transactions = {input}[col].to_numpy().tolist()
        min_support = 100 * {min_support}
        
        patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
        result = [[list(f), s] for f, s in patterns.items()]

        col_item, col_freq = 'itemsets', 'support'
              
        {output} = pd.DataFrame(result, columns=[col_item, col_freq])
        
        # generating rules
        columns = ['Pre-Rule', 'Post-Rule', 'confidence']  
        rules = pyfpgrowth.generate_association_rules(patterns, {min_conf})
        rules = [[list(a), list(b[0]), b[1]] for a, b in rules.items()]
        {rules} = pd.DataFrame(rules, columns=columns) 
        """.format(output=self.output, col=self.column,
                   input=self.named_inputs['input data'],
                   min_support=self.min_support, min_conf=self.confidence,
                   rules=self.rules_output)

        return dedent(code)


class SequenceMiningOperation(Operation):

    MIN_SUPPORT_PARAM = 'min_support'
    ATTRIBUTE_PARAM = 'attribute'
    MAX_LENGTH_PARAM = 'max_length'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:
            if self.MIN_SUPPORT_PARAM not in parameters:
                raise ValueError(
                        _("Parameter '{}' must be informed for task {}").format(
                            self.MIN_SUPPORT_PARAM, self.__class__))

            self.column = parameters.get(self.ATTRIBUTE_PARAM, [''])[0]
            self.output = self.named_outputs.get('output data',
                                                 'output_data_{}'.format(
                                                         self.order))

            self.min_support = float(parameters.get(self.MIN_SUPPORT_PARAM))
            if self.min_support < .0001 or self.min_support > 1.0:
                raise ValueError('Support must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.max_length = abs(int(parameters.get(self.MAX_LENGTH_PARAM,
                                                     10))) or 10
            self.has_import = \
                "from juicer.scikit_learn.library." \
                "prefix_span import PrefixSpan\n"

    def generate_code(self):
        """Generate code."""

        if not len(self.column) > 1:
            self.column = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.column = "'{}'".format(self.column)

        # transactions = [row.tolist() for row in {input}[col].to_numpy().tolist()]
        # transactions = np.array({input}[col].to_numpy().tolist()).tolist()
        code = """
        col = {col}
        transactions = {input}[col].to_numpy().tolist() 
        min_support = {min_support}
        max_length = {max_length}

        span = PrefixSpan(transactions)
        span.run(min_support, max_length)
        result = span.get_patterns()

        {output} = pd.DataFrame(result, columns=['itemsets', 'support'])
        """.format(output=self.output, col=self.column,
                   input=self.named_inputs['input data'],
                   min_support=self.min_support,
                   max_length=self.max_length)

        return dedent(code)


class AssociationRulesOperation(Operation):
    """AssociationRulesOperation.
    """

    MAX_COUNT_PARAM = 'rules_count'
    CONFIDENCE_PARAM = 'confidence'

    ITEMSET_ATTR_PARAM = 'attribute'
    SUPPORT_ATTR_PARAM = 'freq'
    SUPPORT_ATTR_PARAM_VALUE = 'support'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = all([len(named_inputs) == 1,
                             self.contains_results() or len(named_outputs) > 0])
        if self.has_code:
            self.output = named_outputs.get('output data',
                                            'output_data_{}'.format(self.order))

            self.confidence = float(parameters.get(self.CONFIDENCE_PARAM, 0.5))
            if self.confidence < .0001 or self.confidence > 1.0:
                raise ValueError('Confidence must be greater or equal '
                                 'to 0.0001 and smaller than 1.0')

            self.has_import = \
                "from juicer.scikit_learn.library.rules_generator " \
                "import RulesGenerator\n"

            self.support_col = \
                parameters.get(self.SUPPORT_ATTR_PARAM,
                               [self.SUPPORT_ATTR_PARAM_VALUE])[0]
            self.items_col = parameters.get(self.ITEMSET_ATTR_PARAM, [''])[0]
            self.max_rules = parameters.get(self.MAX_COUNT_PARAM, -1) or -1

    def generate_code(self):
        """Generate code."""

        if len(self.items_col) == 0:
            self.items_col = "{input}.columns[0]" \
                .format(input=self.named_inputs['input data'])
        else:
            self.items_col = "'{}'".format(self.items_col)

        code = """
        col_item = {items}
        col_freq = "{freq}"
        
        rg = RulesGenerator(min_conf={min_conf}, max_len={max_len})
        {rules} = rg.get_rules({input}, col_item, col_freq)   
        """.format(min_conf=self.confidence, rules=self.output,
                   input=self.named_inputs['input data'],
                   items=self.items_col, freq=self.support_col,
                   max_len=self.max_rules)

        return dedent(code)
