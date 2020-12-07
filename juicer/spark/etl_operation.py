# -*- coding: utf-8 -*-


import json
import time
from random import random
from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.expression import Expression


class SplitOperation(Operation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
    ('0' means no seed).
    """
    SEED_PARAM = 'seed'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        value = float(parameters.get(self.WEIGHTS_PARAM, 50) or 50)

        self.weights = [value, 100 - value]
        self.seed = parameters.get(self.SEED_PARAM, int(random() * time.time()))
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        self.output1 = self.named_outputs.get(
            'split 1',
            'split_1_task_{}'.format(self.order))

        self.output2 = self.named_outputs.get(
            'split 2',
            'split_2_task_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        # Pay attention: order of ports is reversed in order to port 1
        # be shown in top in interface!
        return sep.join([self.output2, self.output1])

    def generate_code(self):
        code = "{out1}, {out2} = {input}.randomSplit({weights}, {seed})".format(
            out1=self.output1, out2=self.output2,
            input=self.named_inputs['input data'],
            weights=json.dumps(self.weights), seed=self.seed)
        return dedent(code)


class AddRowsOperation(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.parameters = parameters
        self.has_code = any([(len(self.named_inputs) == 2 and len(
            self.named_outputs) > 0), self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'add_rows_{}'.format(self.order))

    def generate_code(self):
        code = "{out} = {input1}.unionAll({input2})".format(
            out=self.output, input1=self.named_inputs['input data 1'],
            input2=self.named_inputs['input data 2'])
        return dedent(code)


class SortOperation(Operation):
    """
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))

    def generate_code(self):
        ascending = []
        attributes = []
        for attr in self.attributes:
            attributes.append(attr['attribute'])
            ascending.append(1 if attr['f'] == 'asc' else 0)

        input_data = self.named_inputs['input data']

        code = "{out} = {input}.orderBy({attrs}, ascending={asc})".format(
            out=self.output, input=input_data, attrs=json.dumps(attributes),
            asc=json.dumps(ascending))

        return dedent(code)


class RemoveDuplicatedOperation(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            self.attributes = []

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'dedup_data_{}'.format(
                                                 self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']

        if self.attributes:
            code = "{output} = {input}.dropDuplicates(subset={attrs})".format(
                output=self.output, input=input_data,
                attrs=json.dumps(self.attributes))
        else:
            code = "{out} = {input}.dropDuplicates()".format(
                out=self.output, input=input_data)
        return dedent(code)


class SampleOrPartitionOperation(Operation):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - withReplacement -> can elements be sampled multiple times
                        (replaced when sampled out)
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
        with replacement: expected number of times each element is chosen;
            fraction must be >= 0
    - seed -> seed for random operation.
    """
    FRACTION_PARAM = 'fraction'
    SEED_PARAM = 'seed'
    WITH_REPLACEMENT_PARAM = 'withReplacement'
    TYPE_PARAM = 'type'
    FOLD_SIZE_PARAM = 'fold_size'
    FOLD_COUNT_PARAM = 'fold_count'
    VALUE_PARAM = 'value'

    TYPE_PERCENT = 'percent'
    TYPE_VALUE = 'value'
    TYPE_HEAD = 'head'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.supports_cache = (self.parameters.get(self.SEED_PARAM) is not None)
        self.seed = parameters.get(self.SEED_PARAM,
                                   int(random() * time.time()))
        self.fold_count = parameters.get(self.FOLD_COUNT_PARAM, 10)
        self.fold_size = parameters.get(self.FOLD_SIZE_PARAM, 1000)
        self.type = parameters.get(self.TYPE_PARAM, self.TYPE_PERCENT)
        self.withReplacement = parameters.get(self.WITH_REPLACEMENT_PARAM,
                                              False)

        if self.type == self.TYPE_PERCENT:
            if self.FRACTION_PARAM in parameters:
                self.fraction = float(parameters[self.FRACTION_PARAM])
                if not (0 <= self.fraction <= 100):
                    msg = _("Parameter '{}' must be in " \
                            "range [0, 100] for task {}") \
                        .format(self.FRACTION_PARAM, __name__)
                    raise ValueError(msg)
                if self.fraction > 1.0:
                    self.fraction *= 0.01
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        self.FRACTION_PARAM, self.__class__))
        elif self.type in [self.TYPE_VALUE, self.TYPE_HEAD]:
            self.value = int(parameters.get(self.VALUE_PARAM, 100))

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'sampled data', 'sampled_data_{}'.format(self.order))

    def generate_code(self):
        code = ''
        input_data = self.named_inputs.get('input data')

        if self.type == self.TYPE_PERCENT:
            code = ("{out} = {input}.sample(withReplacement={wr}, "
                    "fraction={fr}, seed={seed})"
                    .format(out=self.output, input=input_data,
                            wr=self.withReplacement,
                            fr=self.fraction, seed=self.seed))
        elif self.type == self.VALUE_PARAM:
            # Spark 2.0.2 DataFrame API does not have takeSample implemented
            # See [SPARK-15324]
            # This implementation may be inefficient!
            code = ("{out} = {input}.orderBy(functions.rand({seed})).limit("
                    "{limit})".format(out=self.output, input=input_data,
                                      seed=self.seed, limit=self.value))
            pass
        elif self.type == self.TYPE_HEAD:
            code = "{out} = {input}.limit({limit})" \
                .format(out=self.output, input=input_data, limit=self.value)

        return dedent(code)


class IntersectionOperation(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame
    and another data frame.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.parameters = parameters
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'out_{}'.format(self.order))

    def generate_code(self):
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        code = dedent(
            """
            if len({in1}.columns) != len({in2}.columns):
                raise ValueError('{error}')
            {out} = {in1}.intersect({in2})
            """.format(out=self.output, in1=input_data1, in2=input_data2,
                       error=_(
                           'For intersection operation, both input data '
                           'sources must have the same number of attributes '
                           'and types.')))
        return dedent(code)


class DifferenceOperation(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'intersected_data_{}'.format(self.order))

    def generate_code(self):
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        code = "{out} = {in1}.subtract({in2})".format(
            out=self.output, in1=input_data1, in2=input_data2)
        return dedent(code)


class JoinOperation(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'
    ALIASES_PARAM = 'aliases'
    JOIN_PARAMS = 'join_parameters'

    __aslots__ = ('join_parameters', 'join_type', 'new_parameters_format',
            'left_attributes', 'right_attributes', 'aliases', 'output', 
            'match_case', 'keep_right_keys')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False) in [
            'true', 'True', True, '1', 1]

        self.new_parameters_format = self.JOIN_PARAMS in parameters
        self.join_parameters = None

        if self.new_parameters_format: # New parameters format
            self.join_parameters = parameters.get(self.JOIN_PARAMS, {})

            conditions = self.join_parameters.get('conditions', []) or []
            fields_ok = all(['first' in f and 'second' in f for f in conditions])

            self.join_type = self.join_parameters.get('joinType', 'inner') or 'inner'

            if len(conditions) == 0 or not fields_ok:
                    msg = _("Parameter '{}' must be informed for task {}")
                    raise ValueError(msg.format('condition', self.__class__))
        else:
            self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')
            if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                        self.RIGHT_ATTRIBUTES_PARAM in parameters]):
                raise ValueError(
                    _("Parameters '{}' and {} must be informed for task {}").format(
                        self.LEFT_ATTRIBUTES_PARAM, self.RIGHT_ATTRIBUTES_PARAM,
                        self.__class__))
            else:
                self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
                self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)
    
            self.aliases = [
                alias.strip() for alias in
                parameters.get(self.ALIASES_PARAM, 'ds0_, ds1_').split(',')]
    
            if len(self.aliases) != 2:
                raise ValueError(_('You must inform 2 values for alias'))

        self.output = self.named_outputs.get(
            'output data', 'out_data_{}'.format(self.order))

    def generate_code(self):
        if self.new_parameters_format: 
            return self._new_generate_code()
        else:
            return self._legacy_generate_code()
    
    def _get_operator(self, value):
        if value == 'ne':
            return '!='
        elif value == 'gt':
            return '>'
        elif value == 'lt':
            return '<'
        elif value == 'ge':
            return '>='
        elif value == 'le':
            return '<='
        else:
            return '=='

    def _new_generate_code(self):

        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        on_clause = [(f['first'], f['second'], self._get_operator(f.get('op', '='))) 
                for f in self.join_parameters.get('conditions')]

        if self.match_case:
            join_condition = ',\n                '.join(
                [("functions.lower({in0}['{p0}']) {op} "
                  "functions.lower({in1}['{p1}'])").format(
                    in0=input_data1, p0=clause[0], in1=input_data2, p1=clause[1],
                    op=clause[2]) for clause in on_clause])
        else:
            join_condition = ', '.join(
                ["{in0}['{p0}'] == {in1}['{p1}']".format(
                    in0=input_data1, p0=pair[0], in1=input_data2, p1=pair[1])
                    for pair in on_clause])

        code = [dedent("""
            conditions = [
                {cond}
            ]
            result_df = {in0}.join({in1}, on=conditions, how='{how}')""".format(
            cond=join_condition, in0=input_data1,
            in1=input_data2, how=self.join_type))]

        selection_type1 = self.join_parameters.get('firstSelectionType')
        selection_type2 = self.join_parameters.get('secondSelectionType')

        select_code = ''
        if selection_type1 == 3: 
            if selection_type2 == 3:
                raise ValueError(_('No attribute was selected'))
            else:
                code.append("\n{in1}_attrs = []".format(in1=input_data1))

        elif selection_type1 == 2:
            attrs = ["{in1}['{name}'].alias('{alias}')".format(name=attr.get('attribute'), 
                            alias=attr.get('alias'), 
                            in1=input_data1) 
                    for attr in self.join_parameters.get('firstSelect', [])
                    if attr.get('select')]
            code.append(dedent("""
                {in1}_attrs = [
                    {attrs}
                ]
                """.format(in1=input_data1, attrs=',\n                    '.join(attrs))))
        elif selection_type1 == 1:
            code.append(self._code_for_select_all_prefixed(input_data1, 
                self.join_parameters.get('firstPrefix') or 'ds1_'))

        if selection_type2 == 3:
             code.append("\n{in2}_attrs = []".format(in2=input_data2))
        if selection_type2 == 2:
            attrs = ["{in2}['{name}'].alias('{alias}')".format(name=attr.get('attribute'), 
                            alias=attr.get('alias'), 
                            in2=input_data2) 
                    for attr in self.join_parameters.get('secondSelect', [])
                    if attr.get('select')]
            code.append(dedent("""
                {in2}_attrs = [
                    {attrs}
                ]
                """.format(in2=input_data2, attrs=',\n                    '.join(attrs))))
        elif selection_type2 == 1:
            if self.keep_right_keys in ["False", "false", False]:
                remove = [clause[1] for clause in on_clause]
            else:
                remove = None
            code.append(self._code_for_select_all_prefixed(input_data2, 
                self.join_parameters.get('secondPrefix') or 'ds2_', remove))

        code.append(dedent("""
            selected_attrs = {in1}_attrs + {in2}_attrs
            {out} = result_df.select(*selected_attrs)    
        """).format(out=self.output, in1=input_data1, in2=input_data2))

        return dedent('\n'.join(code))

    def _code_for_select_all_prefixed(self, df, prefix, ignore=None):
        if ignore is None:
            ignore = []
        return dedent("""
            {df}_ignore = {ignore}
            {df}_attrs = [{df}[name].alias('{prefix}' + name) 
                for name in {df}.schema.names if name not in {df}_ignore]
        """.format(df=df, prefix=prefix, ignore=repr(ignore)))

    def _legacy_generate_code(self):

        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        on_clause = list(zip(self.left_attributes, self.right_attributes))
        if self.match_case:
            join_condition = ', '.join(
                [("functions.lower(in0_renamed['{a0}{p0}']) == "
                  "functions.lower(in1_renamed['{a1}{p1}'])").format(
                    in0=input_data1, p0=pair[0], in1=input_data2, p1=pair[1],
                    a0=self.aliases[0], a1=self.aliases[1]
                ) for pair in on_clause])
        else:
            join_condition = ', '.join(
                ["in0_renamed['{a0}{p0}'] == in1_renamed['{a1}{p1}']".format(
                    in0=input_data1, p0=pair[0], in1=input_data2, p1=pair[1],
                    a0=self.aliases[0], a1=self.aliases[1])
                    for pair in on_clause])

        code = """
            def _rename_attributes(df, prefix):
                result = df
                for col in df.columns:
                    result = result.withColumnRenamed(col, '{{}}{{}}'.format(
                        prefix, col))
                return result
            in0_renamed = _rename_attributes({in0}, '{a0}')
            in1_renamed = _rename_attributes({in1}, '{a1}')
            condition = [{cond}]
            {out} = in0_renamed.join(
                in1_renamed, on=condition, how='{how}')""".format(
            out=self.output, cond=join_condition, in0=input_data1,
            in1=input_data2, how=self.join_type, a0=self.aliases[0],
            a1=self.aliases[1])

        if self.keep_right_keys in ["False", "false", False]:
            for column in self.right_attributes:
                code += """.drop(in1_renamed['{a1}{col}'])""".format(
                    in1=input_data2, col=column, a1=self.aliases[1])

        return dedent(code)


class DropOperation(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.column = parameters['column']
        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])

    def generate_code(self):
        output = self.named_outputs.get('output data', 'sampled_data_{}'.format(
            self.order))
        input_data = self.named_inputs['input data']
        code = "{out} = {in1}.drop('{drop}')".format(
            out=output, in1=input_data, drop=self.column)
        return dedent(code)


class TransformationOperation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
        replace it.
        - Expression: json describing the transformation expression
    """
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])
        if self.has_code:
            if 'expression' in parameters:
                self.expressions = parameters['expression']
            else:
                msg = _("Parameter must be informed for task {}.")
                raise ValueError(
                    msg.format(self.EXPRESSION_PARAM, self.__class__))
            self.output = self.named_outputs.get(
                'output data', 'sampled_data_{}'.format(self.order))

    def supports_pipeline(self):
        return True

    def generate_code(self):
        input_data = self.named_inputs['input data']
        params = {'input': input_data}

        expr_alias = []
        for expr in self.expressions:
            # Builds the expression and identify the target column
            expression = Expression(expr['tree'], params)
            expr_alias.append("                [{}, '{}']".format(
                expression.parsed_expression, expr['alias']))

        code = dedent("""
            from juicer.spark.ext import CustomExpressionTransformer
            expr_alias = [
                {expr_alias}
            ]
            tmp_out = {in1}
            for expr, alias in expr_alias:
                transformer = CustomExpressionTransformer(outputCol=alias,
                                                          expression=expr)
                tmp_out = transformer.transform(tmp_out)
            {out} = tmp_out
        """.format(out=self.output, in1=input_data,
                   expr_alias=',\n'.join(expr_alias).strip()))

        return dedent(code)


class SelectOperation(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'
    REMOVE_ATTRIBUTES_PARAM = 'remove_attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.output = self.named_outputs.get(
            'output projected data', 'projection_data_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        code = "{out} = {in1}.select({select})".format(
            out=self.output, in1=input_data,
            select=', '.join(['"{}"'.format(x) for x in self.attributes]))
        return dedent(code)


class ReplaceValueOperation(Operation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'
    REPLACEMENT_PARAM = 'replacement'
    VALUE_PARAM = 'value'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, [])
        self.output = self.named_outputs.get(
            'output data', 'replaced_data_{}'.format(self.order))

        if self.REPLACEMENT_PARAM in parameters:
            self.replacement = parameters.get(self.REPLACEMENT_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.REPLACEMENT_PARAM, self.__class__))

        if self.VALUE_PARAM in parameters:
            self.original = parameters.get(self.VALUE_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.VALUE_PARAM, self.__class__))

        def check(v):
            result = False
            try:
                float(v)
                result = True
            except ValueError:
                pass
            result = (result or v.isdigit() or
                      (v[0] in ['\'', '"'] and v[-1] in ['\'', '"']))
            return result

        if not check(self.original):
            raise ValueError(
                _("Parameter '{}' for task '{}' must be a number "
                  "or enclosed in quotes.").format(
                    self.VALUE_PARAM, self.__class__))

        if not check(self.replacement):
            raise ValueError(
                _("Parameter '{}' for task '{}' must be a number "
                  "or enclosed in quotes.").format(
                    self.REPLACEMENT_PARAM, self.__class__))

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        input_data = self.named_inputs['input data']

        code = dedent("""
        try:
            {out} = {in1}.replace({original},
                {replacement}, subset={subset})
        except ValueError as ve:
            if 'Mixed type replacements are not supported' in str(ve):
                raise ValueError('{replacement_same_type}')
            else:
                raise""".format(
            out=self.output, in1=input_data,
            original=self.original,
            replacement=self.replacement,
            replacement_same_type=_('Value and replacement must be of '
                                    'the same type for all attributes'),
            subset=json.dumps(self.attributes)))
        return code


class AggregationOperation(Operation):
    """
    Computes aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """
    ATTRIBUTES_PARAM = 'attributes'
    FUNCTION_PARAM = 'function'
    PIVOT_ATTRIBUTE = 'pivot'
    PIVOT_VALUE_ATTRIBUTE = 'pivot_values'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, [])
        self.functions = parameters.get(self.FUNCTION_PARAM)

        # Attributes are optional
        self.group_all = len(self.attributes) == 0

        if not all([self.FUNCTION_PARAM in parameters, self.functions]):
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.FUNCTION_PARAM, self.__class__))

        for f in parameters[self.FUNCTION_PARAM]:
            if not all([f.get('attribute'), f.get('f'), f.get('alias')]):
                raise ValueError(_('Missing parameter in aggregation function'))

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) == 1, self.contains_results()])
        # noinspection PyArgumentEqualDefault
        self.pivot = next(iter(parameters.get(self.PIVOT_ATTRIBUTE) or []),
                          None)

        self.pivot_values = parameters.get(self.PIVOT_VALUE_ATTRIBUTE)
        self.output = self.named_outputs.get(
            'output data', 'data_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return self.output

    def attribute_traceability(self):
        result = []
        # for i, function in enumerate(self.functions):
        #     if self.pivot:
        #         # FIXME (how to add this when columns are dynamically generated?
        #         pass
        #     else:
        #         result.append(TraceabilityData(
        #             input=self.named_inputs.values()[0],
        #             attribute=function.get('alias', function.get('value')),
        #             derived_from=function['attribute'],
        #             was_value=False))
        return result

    def generate_code(self):
        elements = []
        for i, function in enumerate(self.functions):
            if self.pivot:
                # no alias
                elements.append('''functions.{}('{}')'''.format(
                    function['f'], function['attribute']))
            else:
                elements.append('''functions.{}('{}').alias('{}')'''.format(
                    function['f'], function['attribute'],
                    function.get('alias', function.get('value'))))

        input_data = self.named_inputs['input data']

        if self.pivot:
            if self.pivot_values and self.pivot_values.strip():
                pivot_values = ["{}".format(v.strip()) for v in
                                self.pivot_values.strip().split(',')]
            else:
                pivot_values = None
            pivot_attr = self.pivot
        else:
            pivot_attr = ''
            pivot_values = None

        if not self.group_all:
            group_by = ', '.join(
                ["functions.col('{}')".format(attr)
                 for attr in self.attributes])

            code = dedent("""
                pivot_values = {pivot_values}
                pivot_attr = '{pivot_attr}'
                if pivot_attr:
                    {out} = {input}.groupBy(
                        {key}).pivot(
                            pivot_attr, pivot_values).agg(
                                {el})
                else:
                    {out} = {input}.groupBy(
                        {key}).agg(
                            {el})""".format(
                out=self.output, input=input_data, key=group_by,
                el=', '.join(elements),
                pivot_attr=pivot_attr,
                pivot_values=pivot_values))
        else:
            code = dedent('''
                {output} = {input}.{pivot}agg(
                    {elements})
                '''.format(
                output=self.output, input=input_data,
                elements=', \n        '.join(elements),
                pivot=''))
        return code


class FilterOperation(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'
    ADVANCED_FILTER_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.FILTER_PARAM not in parameters and self.ADVANCED_FILTER_PARAM \
                not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__)))

        self.advanced_filter = parameters.get(self.ADVANCED_FILTER_PARAM) or []
        self.filter = parameters.get(self.FILTER_PARAM) or []

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        params = {'input': input_data}

        filters = [
            "(functions.col('{0}') {1} '{2}')".format(
                f['attribute'], f['f'], f.get('value', f.get('alias')))
            for f in self.filter]

        for expr in self.advanced_filter:
            expression = Expression(expr['tree'], params)
            filters.append('({})'.format(expression.parsed_expression))

        indentation = ' & \n' + (17 * ' ')
        code = """
            {out} = {in1}.filter(
                {f})
            """.format(
            out=self.output, in1=input_data, f=indentation.join(filters))

        return dedent(code)


class SlidingWindowOperation(Operation):
    """
    """
    ATTRIBUTE_PARAM = 'attribute'
    ALIAS_PARAM = 'alias'
    SIZE_PARAM = 'window_size'
    OFFSET_PARAM = 'window_pass'
    HORIZONTAL_PARAM = 'window_gap'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTE_PARAM not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__)))

        self.attribute = (parameters.get(self.ATTRIBUTE_PARAM,  ["col"]))[0]
        self.alias = parameters.get(self.ALIAS_PARAM) or 'attr_'
        self.offset = int(parameters.get(self.OFFSET_PARAM, '0'))
        self.gap = int(parameters.get(self.HORIZONTAL_PARAM, '0'))
        self.size = int(parameters.get(self.SIZE_PARAM, '0')) 

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        params = {'input': input_data}

        code = """
            size = {size}
            gap = {gap}
            win_spec1 = Window.orderBy(functions.lit(1)).rowsBetween(
                Window.unboundedPreceding, Window.currentRow)
            win_spec2 = Window.orderBy(functions.lit(1)).rowsBetween(
                0, size + gap+ -1)
            win_spec3 = Window.orderBy(functions.lit(1)).rowsBetween(
                size+gap, size+gap)
            
            {out} = {in1}.select(functions.row_number().over(win_spec1).alias('row'), 
                      functions.collect_list('{attr}').over(win_spec2).alias('group'), 
                      functions.max('{attr}').over(win_spec3).alias('_tmp_1'))
            if {offset} == 1:
                {out} = {out}.filter(functions.size('group') == size+gap)
            else:
                {out} = {out}.filter(((functions.col('row')) % {offset} == 1) & 
                    (functions.size('group') == size+gap))
            attr_ids = [i for i in range(0, size-1)] + [gap+size-2]
            {out} = {out}.select(
                [(functions.col('group')[id]).alias('{alias}{{}}'.format(id + 1)) 
                    for id in attr_ids])
            """.format(attr=self.attribute, offset=self.offset,
                       alias=self.alias, out=self.output, in1=input_data,
                       size=self.size, gap=self.gap)

        return dedent(code)


class CleanMissingOperation(Operation):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
          * "VALUE": replace by parameter "value",
          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
          * "MEAN": replace by mean value
          * "REMOVE_ROW": remove entire row
          * "REMOVE_COLUMN": remove entire column
        - value: optional, used to replace missing values
    """
    ATTRIBUTES_PARAM = 'attributes'
    CLEANING_MODE_PARAM = 'cleaning_mode'
    VALUE_PARAMETER = 'value'
    MIN_MISSING_RATIO_PARAM = 'min_missing_ratio'
    MAX_MISSING_RATIO_PARAM = 'max_missing_ratio'

    VALUE = 'VALUE'
    MEAN = 'MEAN'
    MODE = 'MODE'
    MEDIAN = 'MEDIAN'
    REMOVE_ROW = 'REMOVE_ROW'
    REMOVE_COLUMN = 'REMOVE_COLUMN'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__)))
        self.cleaning_mode = parameters.get(self.CLEANING_MODE_PARAM,
                                            self.REMOVE_ROW)

        self.value = parameters.get(self.VALUE_PARAMETER)

        self.min_missing_ratio = parameters.get(self.MIN_MISSING_RATIO_PARAM)
        self.max_missing_ratio = parameters.get(self.MAX_MISSING_RATIO_PARAM)

        # In this case, nothing will be generated besides create reference to
        # data frame
        self.has_code = any([all([
            any([self.value is not None, self.cleaning_mode != self.VALUE]),
            len(self.named_inputs) > 0]), self.contains_results()])
        self.output = self.named_outputs.get('output result',
                                             'out_{}'.format(self.order))

    def generate_code(self):

        input_data = self.named_inputs['input data']

        pre_code = []
        partial = []
        attrs_json = json.dumps(self.attributes)

        if any([self.min_missing_ratio, self.max_missing_ratio]):
            self.min_missing_ratio = float(self.min_missing_ratio)
            self.max_missing_ratio = float(self.max_missing_ratio)

            # Based on http://stackoverflow.com/a/35674589/1646932
            select_list = [
                ("\n    (functions.avg(functions.col('{0}').isNull()."
                 "cast('int'))).alias('{0}')").format(attr)
                for attr in self.attributes]
            pre_code.extend([
                "# Computes the ratio of missing values for each attribute",
                "ratio_{0} = {0}.select({1}).collect()".format(
                    input_data, ', '.join(select_list)), "",
                "attributes_{0} = [c for c in {1} "
                "\n                  if {2} <= ratio_{0}[0][c] <= {3}]".format(
                    input_data, attrs_json, self.min_missing_ratio,
                    self.max_missing_ratio)
            ])
        else:
            pre_code.append(
                "attributes_{0} = {1}".format(input_data, attrs_json))

        if self.cleaning_mode == self.REMOVE_ROW:
            partial.append("""
                {0} = {1}.na.drop(how='any', subset=attributes_{1})""".format(
                self.output, input_data))

        elif self.cleaning_mode == self.VALUE:
            # value = ast.literal_eval(self.value)
            partial.append(
                "\n    {0} = {1}.na.fill(value={2}, "
                "subset=attributes_{1})".format(self.output, input_data,
                                                self.value))

        elif self.cleaning_mode == self.REMOVE_COLUMN:
            # Based on http://stackoverflow.com/a/35674589/1646932"
            partial.append(
                "\n{0} = {1}.select("
                "[c for c in {1}.columns if c not in attributes_{1}])".format(
                    self.output, input_data))

        elif self.cleaning_mode == self.MODE:
            # Based on http://stackoverflow.com/a/36695251/1646932
            # But null values cause exception, so it needs to remove them
            partial.append("""
                import decimal
                md_replace_{1} = dict()
                for md_attr_{1} in attributes_{1}:
                    md_count_{1} = {1}.na.drop(subset=[md_attr_{1}]).groupBy(
                        md_attr_{1}).count().orderBy(
                            functions.desc('count')).limit(1)
                    # Spark does not support BigDecimal!
                    if isinstance(md_count_{1}.collect()[0][0],
                        decimal.Decimal):
                        replacement = float(md_count_{1}.collect()[0][0])
                    else:
                        replacement = md_count_{1}.collect()[0][0]
                    md_replace_{1}[md_attr_{1}] = replacement
                {0} = {1}.fillna(value=md_replace_{1})""".format(
                self.output, input_data)
            )

        elif self.cleaning_mode == self.MEDIAN:
            # See http://stackoverflow.com/a/31437177/1646932
            # But null values cause exception, so it needs to remove them
            partial.append("""
                mdn_replace_{1} = dict()
                for mdn_attr_{1} in attributes_{1}:
                    # Computes median value for column with relat. error=10%
                    mdn_{1} = {1}.na.drop(subset=[mdn_attr_{1}])\\
                        .approxQuantile(str(mdn_attr_{1}), [.5], .1)
                    mdn_replace_{1}[mdn_attr_{1}] = mdn_{1}[0]
                {0} = {1}.fillna(value=mdn_replace_{1})""".format(
                self.output, input_data))

        elif self.cleaning_mode == self.MEAN:
            partial.append("""
                avg_{1} = {1}.select([functions.avg(c).alias(c)
                                        for c in attributes_{1}]).collect()
                # Convert to float because Spark complains about Decimal
                values_{1} = dict([(c, float(avg_{1}[0][c]))
                    for c in attributes_{1}])
                {0} = {1}.na.fill(value=values_{1})""".format(self.output,
                                                              input_data))
        else:
            raise ValueError(
                _("Parameter '{}' has an incorrect value '{}' in {}").format(
                    self.CLEANING_MODE_PARAM, self.cleaning_mode,
                    self.__class__))

        return '\n'.join(pre_code) + \
               "\nif len(attributes_{0}) > 0:".format(input_data) + \
               '\n    '.join([dedent(line) for line in partial]).replace(
                   '\n',
                   '\n    ') + \
               "\nelse:\n    {0} = {1}".format(self.output, input_data)


class AddColumnsOperation(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    Implementation based on post http://stackoverflow.com/a/40510320/1646932
    """
    ALIASES_PARAM = 'aliases'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) == 2, self.contains_results()])
        self.aliases = [
            alias.strip() for alias in
            parameters.get(self.ALIASES_PARAM, 'ds0_, ds1_').split(',')]

        if len(self.aliases) != 2:
            raise ValueError('You must inform 2 values for alias')

        self.output = self.named_outputs.get(
            'output data', 'add_col_data_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        code = """

            def _add_column_index(df, prefix):
                # Create new attribute names
                old_attrs = ['{{}}{{}}'.format(prefix, name)
                    for name in df.schema.names]
                new_attrs = old_attrs + ['_inx']

                # Add attribute index
                return df.rdd.zipWithIndex().map(
                    lambda row_inx: row_inx[0] + (row_inx[1],)).toDF(new_attrs)

            input1_indexed = _add_column_index({input1}, '{a1}')
            input2_indexed = _add_column_index({input2}, '{a2}')

            {out} = input1_indexed.join(
                input2_indexed,
                input1_indexed._inx == input2_indexed._inx,
                'inner').drop(input1_indexed._inx).drop(input2_indexed._inx)
            """.format(input1=input_data1, input2=input_data2, out=self.output,
                       a1=self.aliases[0], a2=self.aliases[1])
        return dedent(code)


class PivotTableOperation(Operation):
    AGGREGATION_ATTRIBUTES_PARAM = 'aggregation_attributes'
    PIVOT_ATTRIBUTE_PARAM = 'pivot'
    FUNCTIONS_PARAM = 'functions'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if not all([self.AGGREGATION_ATTRIBUTES_PARAM in parameters,
                    self.PIVOT_ATTRIBUTE_PARAM in parameters,
                    self.FUNCTIONS_PARAM in parameters]):
            raise ValueError(
                _("Required parameters must be informed for task {}".format(
                    self.__class__)))

        self.aggregation_attributes = parameters.get(
            self.AGGREGATION_ATTRIBUTES_PARAM, [])
        self.pivot = parameters.get(self.PIVOT_ATTRIBUTE_PARAM)
        self.functions = parameters.get(self.FUNCTIONS_PARAM)

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        elements = []
        for i, function in enumerate(self.functions):
            elements.append('''functions.{}('{}').alias('{}')'''.format(
                function['f'].lower(), function['attribute'],
                function['alias']))

        input_data = self.named_inputs['input data']

        group_by = ', '.join(
            ["functions.col('{}')".format(attr)
             for attr in self.aggregation_attributes])
        pivot = "functions.col('{}')".format(self.pivot)

        code = """
            {out} = {input}.groupBy({keys}).pivot({pivot}).agg({el})
            """.format(out=self.output, input=input_data, keys=group_by,
                       pivot=pivot,
                       el=', \n        '.join(elements))

        return dedent(code)


class ExecutePythonOperation(Operation):
    PYTHON_CODE_PARAM = 'code'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if not all([self.PYTHON_CODE_PARAM in parameters]):
            msg = _("Required parameter {} must be informed for task {}")
            raise ValueError(msg.format(self.PYTHON_CODE_PARAM, self.__class__))

        self.code = parameters.get(self.PYTHON_CODE_PARAM)

        # Always execute
        self.has_code = True
        self.out1 = self.named_outputs.get('output data 1',
                                           'out_1_{}'.format(self.order))
        self.out2 = self.named_outputs.get('output data 2',
                                           'out_2_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        return sep.join([self.out1, self.out2])

    def generate_code(self):
        in1 = self.named_inputs.get('input data 1', 'None')

        in2 = self.named_inputs.get('input data 2', 'None')

        code = dedent("""
        import json
        from RestrictedPython.Guards import safe_builtins
        from RestrictedPython.compile import compile_restricted
        from RestrictedPython.PrintCollector import PrintCollector

        results = [r[1].result() for r in task_futures.items() if r[1].done()]
        results = dict([(r['task_name'], r) for r in results])
        # Input data
        in1 = {in1}
        in2 = {in2}

        # Output data, initialized as None
        out1 = None
        out2 = None

        # Variables and language supported
        ctx = {{
            'createDataFrame': spark_session.createDataFrame,
            'wf_results': results,
            'in1': in1,
            'in2': in2,
            'out1': out1,
            'out2': out2,
            'VectorAssembler': VectorAssembler, # from pyspark.ml.feature
            'to_dense_vector': functions.udf(
                lambda vs: Vectors.dense(vs), VectorUDT()),

            # Restrictions in Python language
             '_write_': lambda v: v,
            '_getattr_': getattr,
            '_getitem_': lambda ob, index: ob[index],
            '_getiter_': lambda it: it,
            '_print_': PrintCollector,
            'json': json,
        }}
        user_code = \"\"\"{code}\"\"\"

        ctx['__builtins__']= safe_builtins

        compiled_code = compile_restricted(user_code,
        str('python_execute_{order}'), str('exec'))
        try:
            exec(compiled_code, ctx)

            # Retrieve values changed in the context
            out1 = ctx['out1']
            out2 = ctx['out2']

            if '_print' in ctx:
                emit_event(name='update task',
                    message=ctx['_print'](),
                    status='RUNNING',
                    identifier='{id}')
        except NameError as ne:
            raise ValueError('{msg1}'.format(ne))
        except ImportError as ie:
            raise ValueError(_('Command import is not supported'))
        """.format(in1=in1, in2=in2, code=self.code.encode(
            'unicode_escape').decode('utf8'),
                   name="execute_python", order=self.order,
                   msg1=_('Invalid name: {}. '
                          'Many Python commands are not available in Lemonade'),
                   id=self.parameters['task']['id']))
        # code += "\n# -- BEGIN user code\n{code}\n# -- END user code\n".format(
        #    code=dedent(self.code))

        code += dedent("""
        {out1} = out1
        {out2} = out2
        """.format(out1=self.out1, out2=self.out2))
        return dedent(code)


class ExecuteSQLOperation(Operation):
    QUERY_PARAM = 'query'
    NAMES_PARAM = 'names'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if not all([self.QUERY_PARAM in parameters]):
            msg = _("Required parameter {} must be informed for task {}")
            raise ValueError(msg.format(self.QUERY_PARAM, self.__class__))

        self.query = ExecuteSQLOperation._escape_string(
            parameters.get(self.QUERY_PARAM).strip().replace('\n', ' '))
        if self.query[:6].upper() != 'SELECT':
            raise ValueError(_('Invalid query. Only SELECT is allowed.'))

        if self.NAMES_PARAM in parameters:
            self.names = [
                n.strip() for n in parameters.get(self.NAMES_PARAM).split(',')
                if n.strip()]
        else:
            self.names = None

        self.has_code = any([len(self.named_outputs) > 0,
                             self.contains_results()])
        self.input1 = self.named_inputs.get('input data 1')
        self.input2 = self.named_inputs.get('input data 2')
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    @staticmethod
    def _escape_string(value):
        """ Escape a SQL string. Borrowed from
        https://github.com/PyMySQL/PyMySQL/blob/master/pymysql/converters.py"""
        return value
        # _escape_table = [unichr(x) for x in range(128)]
        # _escape_table[0] = u'\\0'
        # _escape_table[ord('\\')] = u'\\\\'
        # _escape_table[ord('\n')] = u'\\n'
        # _escape_table[ord('\r')] = u'\\r'
        # _escape_table[ord('\032')] = u'\\Z'
        # _escape_table[ord('"')] = u'\\"'
        # _escape_table[ord("'")] = u"\\'"
        # return value.translate(_escape_table)

    def generate_code(self):
        code = dedent("""
        from pyspark.sql import SQLContext

        # Input data
        sql_context = SQLContext(spark_session.sparkContext)
        if {in1} is not None:
            sql_context.registerDataFrameAsTable({in1}, 'ds1')
        if {in2} is not None:
            sql_context.registerDataFrameAsTable({in2}, 'ds2')
        query = {query}
        {out} = sql_context.sql(query)
        names = {names}
        if names is not None and len(names) > 0:
            old_names = {out}.schema.names
            if len(old_names) != len(names):
                raise ValueError('{invalid_names}')
            rename = [functions.col(pair[0]).alias(pair[1])
                for pair in zip(old_names, names)]
            {out} = {out}.select(*rename)
        """.format(in1=self.input1, in2=self.input2, query=repr(self.query),
                   out=self.output, names=repr(self.names),
                   invalid_names=_('Invalid names. Number of attributes in '
                                   'result differs from names informed.')))
        return code


class TableLookupOperation(Operation):
    """
    Allow lookup a value in a lookup table.
    In the case of Apache Spark, the lookup table is a small data set that is
    broadcast to all processing nodes
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = False

    def generate_code(self):
        pass


class WindowTransformationOperation(Operation):
    """
    Returns a new DataFrame applying transformations over a window.
    See documentation about window operations in Spark.
    """
    RANKING_FUNCTIONS = ['rank', 'dense_rank', 'percent_rank', 'ntile',
                         'row_number']
    PARTITION_ATTRIBUTE_PARAM = 'partition_attribute'
    ORDER_BY_PARAM = 'order_by'
    ROWS_FROM_PARAM = 'rows_from'
    ROWS_TO_PARAM = 'rows_to'
    RANGE_FROM_PARAM = 'range_from'
    RANGE_TO_PARAM = 'range_to'
    EXPRESSIONS_PARAM = 'expressions'
    TYPE_PARAM = 'type'

    ROWS = 'rows'
    RANGE = 'range'
    NONE = 'none'
    TYPES = (ROWS, RANGE, NONE)

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
            [len(self.named_inputs) > 0, self.contains_results()])

        self.partition_attribute = None
        self.expressions = None
        if self.has_code:
            required_params = [self.EXPRESSIONS_PARAM,
                               self.PARTITION_ATTRIBUTE_PARAM]
            for required in required_params:
                if required in parameters:
                    setattr(self, required, parameters[required])
                else:
                    msg = _("Parameter '{}' must be informed for task {}")
                    raise ValueError(msg.format(required, self.__class__))

            if isinstance(self.partition_attribute, list):
                self.partition_attribute = self.partition_attribute[0]
            self.type = parameters.get(self.TYPE_PARAM, self.NONE)
            if self.type not in self.TYPES:
                msg = _('The value for parameter {} must be one of these: {}.')
                raise ValueError(msg.format('type', ', '.join(self.TYPES)))

            for expression in self.expressions:
                if 'alias' not in expression:
                    msg = _("Parameter '{}' must be informed in all "
                            "expressions in task {}.")
                    raise ValueError(msg.format('alias', self.__class__))
                if 'tree' not in expression:
                    msg = _("Parameter '{}' must be informed in all "
                            "expressions in task {}.")
                    raise ValueError(msg.format('tree', self.__class__))

            self.order_by = parameters.get(self.ORDER_BY_PARAM, [])
            self.rows_from = parameters.get(
                self.ROWS_FROM_PARAM,
                'Window.unboundedPreceding') or 'Window.unboundedPreceding'
            self.rows_to = parameters.get(
                self.ROWS_TO_PARAM,
                'Window.unboundedFollowing') or 'Window.unboundedFollowing'

            self.range_from = parameters.get(
                self.RANGE_FROM_PARAM,
                'Window.unboundedPreceding') or 'Window.unboundedPreceding'
            self.range_to = parameters.get(
                self.RANGE_TO_PARAM,
                'Window.unboundedFollowing') or 'Window.unboundedFollowing'

            self.output = self.named_outputs.get(
                'output data', 'data_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        input_data = self.named_inputs['input data']

        attributes = []
        for attr in self.order_by:
            if attr['f'] == 'asc':
                attributes.append(
                    "functions.asc('{}')".format(attr['attribute']))
            else:
                attributes.append(
                    "functions.desc('{}')".format(attr['attribute']))

        if attributes:
            order_by_code = ".orderBy({attrs})".format(
                attrs=', '.join(attributes))
        else:
            order_by_code = ''

        rows_code = ''
        range_code = ''
        if self.type == self.ROWS:
            from_int = None
            # noinspection PyUnresolvedReferences
            if self.rows_from.isdigit():
                from_int = -abs(int(self.rows_from))
                self.rows_from = from_int

            to_int = None
            if self.rows_to.isdigit():
                to_int = int(self.rows_to)

            if to_int and from_int and from_int > to_int:
                raise ValueError(_(
                    'Start of offset must be >= its end: [{}, {}]').format(
                    self.rows_from, self.rows_to))
            rows_code = ".rowsBetween({rows_from}, {rows_to})".format(
                rows_from=self.rows_from, rows_to=self.rows_to)
        elif self.type == self.RANGE:
            range_code = ".rangeBetween({range_from}, {range_to})".format(
                range_from=self.range_from, range_to=self.range_to)

        aliases, new_attrs, params = self._get_expressions()

        code = dedent("""
            from juicer.spark.ext import CustomExpressionTransformer

            rank_window = Window.partitionBy('{partition}'){order_by_code}
            window = Window.partitionBy(
                '{partition}'){order_by_code}{rows_code}{range_code}
            specs = [{new_attrs}]
            aliases = {aliases}

            {out} = {in1}
            for i, spec in enumerate(specs):
                {out} = {out}.withColumn(aliases[i], spec)
        """.format(out=self.output, in1=input_data,
                   partition=self.partition_attribute,
                   order_by_code=order_by_code,
                   rows_code=rows_code,
                   range_code=range_code,
                   new_attrs=',\n                     '.join(new_attrs),
                   aliases=json.dumps(aliases)))

        return dedent(code)

    def _get_expressions(self):
        aliases = []
        new_attrs = []
        params = {}
        for expr in self.expressions:
            if expr['tree'].get('callee', {}).get(
                    'name') not in self.RANKING_FUNCTIONS:
                params['window'] = 'window'
            else:
                params['window'] = 'rank_window'

            expression = Expression(expr['tree'], params, window=True)
            built_expr = expression.parsed_expression
            new_attrs.append(built_expr)
            aliases.append(expr['alias'])
        return aliases, new_attrs, params


class SplitKFoldOperation(Operation):
    """
    K-fold cross-validation  partitioned dataset into k equal sized
    subsamples called folds, it can be randomly or stratified.
    """
    K_FOLD_PARAM = 'k_fold'
    TYPE_PARAM = 'type'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    ALIAS_PARAM = 'alias_fold'
    SEED_PARAM = 'seed'

    TYPE_RANDOM = 'random'
    TYPE_STRATIFIED = 'stratified'
    TYPE_STRATIFIED_EXACT = 'stratified_exact'
    TYPE_RANDOM_EXACT = 'random_exact'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.K_FOLD_PARAM in parameters:
            self.k_fold = int(parameters.get(self.K_FOLD_PARAM, 10))
            if self.k_fold < 2:
                raise ValueError(_("Parameter '{}' informed for task '{}' \
                                   must be at >= 2").format(self.K_FOLD_PARAM,
                                                            self.__class__))
        else:
            raise ValueError(_("Parameter '{}' must be informed \
                                for task {}").format(self.K_FOLD_PARAM,
                                                     self.__class__))

        self.type = parameters.get(self.TYPE_PARAM, self.TYPE_RANDOM)
        self.label = (parameters.get(self.LABEL_PARAM) or ['label'])[0]
        self.alias = parameters.get(self.ALIAS_PARAM, 'fold')
        self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))

    def generate_code(self):
        code = ""
        input_data = self.named_inputs.get('input data')
        display_text = self.parameters['task']['forms'].get(
            'display_text', {'value': 1}).get('value', 1) in (1, '1')

        if self.type == self.TYPE_RANDOM:
            code = ("""
                seed = {seed}
                k = {k_fold}
                weights = [1.0] * k # Spark normalizes weights to sum up to 1.0
                split_dfs = {input}.randomSplit(weights, seed)
                final_split = []
                for i, split_df in enumerate(split_dfs):
                    final_split.append(split_df.withColumn(
                        '{alias}', functions.lit(i)))
                {out} = functools.reduce(DataFrame.union, final_split)
                # Emit a summary of partition
                display_text = {display_text}
                if display_text:
                    summary = {out}.groupBy('{alias}').count().orderBy(
                        '{alias}')
                    dataframe_util.emit_sample(
                        '{task_id}', summary, emit_event, None,
                        title='{title}')
                """.format(input=input_data,
                           out=self.output,
                           task_id=self.parameters['task_id'],
                           title=_('Summary for K-fold'),
                           display_text=display_text,
                           k_fold=self.k_fold,
                           alias=self.alias,
                           seed=self.seed))
        elif self.type == self.TYPE_RANDOM_EXACT:
            code = ("""
                def get_split_bounds(l, n):
                    return [[(i*l)//n, ((i+1)*l)//n - 1] for i in range(n)]

                seed = {seed}
                k = {k_fold}

                total = {input}.count()
                # orderBy is required
                w = Window().orderBy(functions.rand(seed=seed))
                df = {input}.withColumn(
                    'row_num_tmp', functions.row_number().over(w))

                bounds = get_split_bounds(total, k)
                when_col = None
                for i, bound in enumerate(bounds):
                    condition = (df.row_num_tmp - 1 >= bound[0]) & (
                        df.row_num_tmp - 1 <= bound[1])
                    if when_col is None:
                        when_col = functions.when(condition, i)
                    else:
                        when_col = when_col.when(condition, i)
                df = df.withColumn('{alias}', when_col)
                {out} = df.drop('row_num_tmp')

                # Emit a summary of partition
                display_text = {display_text}
                if display_text:
                    summary = {out}.groupBy('{alias}').count().orderBy(
                        '{alias}')
                    dataframe_util.emit_sample(
                        '{task_id}', summary, emit_event, None,
                        title='{title}')
                """.format(input=input_data,
                           out=self.output,
                           task_id=self.parameters['task_id'],
                           title=_('Summary for K-fold'),
                           display_text=display_text,
                           k_fold=self.k_fold,
                           alias=self.alias,
                           seed=self.seed))
        elif self.type == self.TYPE_STRATIFIED:

            # Stratification seeks to ensure that each fold is
            # representative of all strata of the data. Generally
            # this is done in a supervised way for classification
            # and aims to ensure each class is (approximately) equally
            # represented across each test fold (which are of course
            # combined in a complementary way to form training folds).

            code = ("""
                seed = {seed}
                k = {k_fold}
                weights = [1.0] * k # Spark normalizes weights to sum up to 1.0
                distinct_labels = [row[0] for row in
                    {input}.select('{label}').distinct().collect()]

                # We will have a matrix MxK where M is the number of distinct
                # labels and K is the number of folds
                split_df_by_label = [
                    {input}.where(functions.col('{label}').eqNullSafe(
                        label)).randomSplit(weights, seed)
                        for label in distinct_labels]

                # Union of all datasets, partitioned by label and split in
                # k folds
                {out} = None
                for df_by_label in split_df_by_label:
                    for i, df_by_split in enumerate(df_by_label):
                        df_by_split = df_by_split.withColumn(
                            '{alias}', functions.lit(i))
                        if {out} is None:
                            {out} = df_by_split
                        else:
                            {out} = {out}.union(df_by_split)

                # Emit a summary of partition
                display_text = {display_text}
                if display_text:
                    summary = {out}.groupBy(
                        '{label}', '{alias}').count().orderBy(
                            '{label}', '{alias}')
                    dataframe_util.emit_sample(
                        '{task_id}', summary, emit_event, None,
                        title='{title}')
            """.format(input=input_data,
                       out=self.output,
                       k_fold=self.k_fold,
                       display_text=display_text,
                       label=self.label,
                       alias=self.alias,
                       seed=self.seed,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_('Summary for K-fold')))

        elif self.type == self.TYPE_STRATIFIED_EXACT:
            code = ("""
                def get_fold_number(k, count, row_num):
                    for i in range(k):
                        if (i*count)//k <= row_num - 1 <= ((i+1)*count)//k - 1:
                            return i

                udf_fold = functions.udf(get_fold_number, types.IntegerType())

                seed = {seed}

                # Requires 2 window specifications
                # First to add a row number inside group and second to count
                # the total of rows inside the group
                w1 = Window().partitionBy('{label}').orderBy(
                    functions.rand(seed=seed)).rowsBetween(
                        Window.unboundedPreceding, Window.currentRow)
                w2 = Window().partitionBy('{label}').rowsBetween(
                        Window.unboundedPreceding, Window.unboundedFollowing)

                df = {input}.withColumn(
                    'row_num_tmp', functions.row_number().over(w1)).withColumn(
                    'count_tmp', functions.count('*').over(w2))

                df = df.withColumn('{alias}',
                    udf_fold(
                        functions.lit({k_fold}), df.count_tmp, df.row_num_tmp))

                {out} = df.drop('count_tmp', 'row_num_tmp')
                # Emit a summary of partition
                display_text = {display_text}
                if display_text:
                    summary = {out}.groupBy(
                        '{label}', '{alias}').count().orderBy(
                            '{label}', '{alias}')
                    dataframe_util.emit_sample(
                        '{task_id}', summary, emit_event, None,
                        title='{title}')
            """.format(input=input_data,
                       out=self.output,
                       display_text=display_text,
                       k_fold=self.k_fold,
                       label=self.label,
                       alias=self.alias,
                       seed=self.seed,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_('Summary for K-fold')))

        return dedent(code)
