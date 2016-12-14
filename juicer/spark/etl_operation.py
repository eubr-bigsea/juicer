import ast
import json
import time
from random import random
from textwrap import dedent

import sys

from juicer.spark.expression import Expression
from juicer.spark.operation import Operation


class RandomSplit(Operation):
    """
    Randomly splits the Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
    ('0' means no seed).
    """
    SEED_PARAM = 'seed'
    WEIGHTS_PARAM = 'weights'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        value = float(parameters.get(self.WEIGHTS_PARAM, 50))
        self.weights = [value, 100 - value]
        self.seed = parameters.get(self.SEED_PARAM, int(random() * time.time()))
        self.has_code = len(self.outputs) > 0

    def generate_code(self):
        if len(self.inputs) == 1:
            output1 = self.outputs[0] if len(
                self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            output2 = self.outputs[1] if len(self.outputs) == 2 else '_'

            code = """{0}, {1} = {2}.randomSplit({3}, {4})""".format(
                output1, output2, self.inputs[0],
                json.dumps(self.weights), self.seed)
        else:
            code = ""
        return dedent(code)


class AddRows(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.parameters = parameters

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        if len(self.inputs) == 2:
            code = "{0} = {1}.unionAll({2})".format(output,
                                                    self.inputs[0],
                                                    self.inputs[1])
        else:
            code = ""
        return dedent(code)


class Sort(Operation):
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

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        ascending = []
        attributes = []
        for attr in self.attributes:
            attributes.append(attr['attribute'])
            ascending.append(1 if attr['f'] == 'asc' else 0)

        code = "{0} = {1}.orderBy({2}, \n            ascending={3})".format(
            output, self.inputs[0], json.dumps(attributes),
            json.dumps(ascending))

        return dedent(code)


class Distinct(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            self.attributes = []

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        if len(self.inputs) == 1:
            if self.attributes:
                code = "{} = {}.dropDuplicates(subset={})".format(
                    output, self.inputs[0], json.dumps(self.attributes))
            else:
                code = "{} = {}.dropDuplicates()".format(output, self.inputs[0])
        else:
            code = ""

        return dedent(code)


class SampleOrPartition(Operation):
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

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

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
                    msg = "Parameter '{}' must be in " \
                          "range [0, 100] for task {}" \
                        .format(self.FRACTION_PARAM, __name__)
                    raise ValueError(msg)
                if self.fraction > 1.0:
                    self.fraction *= 0.01
            else:
                raise ValueError(
                    "Parameter '{}' must be informed for task {}".format(
                        self.FRACTION_PARAM, self.__class__))
        elif self.type in [self.TYPE_VALUE, self.TYPE_HEAD]:
            self.value = int(parameters.get(self.VALUE_PARAM, 100))
        else:
            raise ValueError(
                "Invalid type '{}' for task {}".format(
                    self.type, self.__class__))

        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        code = ''
        if self.type == self.TYPE_PERCENT:
            code = ("{} = {}.sample(withReplacement={}, fraction={}, seed={})"
                    .format(self.output, self.inputs[0],
                            self.withReplacement,
                            self.fraction, self.seed))
        elif self.type == self.VALUE_PARAM:
            # Spark 2.0.2 DataFrame API does not have takeSample implemented
            # See [SPARK-15324]
            # This implementation may be innefficient!
            code = ("{} = {}.sample(withReplacement={}, "
                    "fraction={}, seed={}).limit({})"
                    .format(self.output, self.inputs[0],
                            self.withReplacement,
                            1.0, self.seed, self.value))
            pass
        elif self.type == self.TYPE_HEAD:
            code = "{} = {}.limit({})" \
                .format(self.output, self.inputs[0], self.value)

        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame
    and another frame.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.parameters = parameters

    def generate_code(self):
        if len(self.inputs) == 2:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])

            code = "{} = {}.intersect({})".format(
                output, self.inputs[0], self.inputs[1])
        else:
            code = ''
        return dedent(code)


class Difference(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

    def generate_code(self):
        code = "{} = {}.subtract({})".format(
            self.outputs[0], self.inputs[0], self.inputs[1])
        return dedent(code)


class Join(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.keep_right_keys = parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.LEFT_ATTRIBUTES_PARAM, self.RIGHT_ATTRIBUTES_PARAM,
                    self.__class__))
        else:
            self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
            self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

    def generate_code(self):
        on_clause = zip(self.left_attributes, self.right_attributes)
        join_condition = ', '.join([
                                       '{}.{} == {}.{}'.format(self.inputs[0],
                                                               pair[0],
                                                               self.inputs[1],
                                                               pair[1]) for pair
                                       in on_clause])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        code = """
            cond_{0} = [{1}]
            {0} = {2}.join({3}, on=cond_{0}, how='{4}')""".format(
            output, join_condition, self.inputs[0], self.inputs[1],
            self.join_type)

        # TO-DO: Convert str False to boolean for evaluation
        if self.keep_right_keys == "False":
            for column in self.right_attributes:
                code += """.drop({}.{})""".format(self.inputs[1], column)

        return dedent(code)


class Drop(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.column = parameters['column']

    def generate_code(self):
        code = """{} = {}.drop('{}')""".format(
            self.outputs[0], self.inputs[0], self.column)
        return dedent(code)


class Transformation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
        replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if all(['alias' in parameters, 'expression' in parameters]):
            self.alias = parameters['alias']
            self.json_expression = json.loads(parameters['expression'])['tree']
        else:
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.ALIAS_PARAM, self.EXPRESSION_PARAM, self.__class__))

    def generate_code(self):
        if len(self.inputs) > 0:
            # Builds the expression and identify the target column
            expression = Expression(self.json_expression)
            built_expression = expression.parsed_expression
            if len(self.outputs) > 0:
                output = self.outputs[0]
            else:
                output = '{}_tmp'.format(self.inputs[0])

            # Builds the code
            code = """{} = {}.withColumn('{}', {})""".format(output,
                                                             self.inputs[0],
                                                             self.alias,
                                                             built_expression)
        else:
            code = ''
        return dedent(code)


class Select(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        code = """{} = {}.select({})""".format(
            output, self.inputs[0],
            ', '.join(['"{}"'.format(x) for x in self.attributes]))
        return dedent(code)


class Aggregation(Operation):
    """
    Compute aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """
    ATTRIBUTES_PARAM = 'attributes'
    FUNCTION_PARAM = 'function'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        self.functions = parameters.get(self.FUNCTION_PARAM)

        if not all([self.ATTRIBUTES_PARAM in parameters,
                    self.FUNCTION_PARAM in parameters,
                    self.attributes, self.functions]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.FUNCTION_PARAM, self.__class__))
        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        elements = []
        for i, function in enumerate(self.functions):
            elements.append('''{}('{}').alias('{}')'''.format(
                function['f'].lower(), function['attribute'],
                function['alias']))

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        group_by = ', '.join(
            ["col('{}')".format(attr) for attr in self.attributes])

        code = '''{} = {}.groupBy({}).agg(\n        {})'''.format(
            output, self.inputs[0], group_by, ', \n        '.join(elements))
        return dedent(code)


class Filter(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.FILTER_PARAM not in parameters:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__))

        self.filter = parameters.get(self.FILTER_PARAM)

        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        filters = [
            "(col('{0}') {1} '{2}')".format(f['attribute'], f['f'], f['value'])
            for f in self.filter]

        code = """{} = {}.filter({})""".format(
            output, self.inputs[0], ' & '.join(filters))
        return dedent(code)


class CleanMissing(Operation):
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

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.cleaning_mode = parameters.get(self.CLEANING_MODE_PARAM,
                                            self.REMOVE_ROW)

        self.value = parameters.get(self.VALUE_PARAMETER)

        self.min_missing_ratio = float(
            parameters.get(self.MIN_MISSING_RATIO_PARAM, 0))
        self.max_missing_ratio = float(
            parameters.get(self.MAX_MISSING_RATIO_PARAM, 1))

        # In this case, nothing will be generated besides create reference to
        # data frame
        if (self.value is None and self.cleaning_mode == self.VALUE) or len(
                self.inputs) == 0:
            self.has_code = False

    def generate_code(self):
        if not self.has_code:
            return "{} = {}".format(self.outputs[0], self.inputs[0])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        pre_code = []
        partial = []
        attrs_json = json.dumps(self.attributes)

        if any([self.min_missing_ratio, self.max_missing_ratio]):
            self.min_missing_ratio = self.min_missing_ratio or 0.0
            self.max_missing_ratio = self.max_missing_ratio or 1.0

            # Based on http://stackoverflow.com/a/35674589/1646932
            select_list = [
                "\n    (count('{0}') / count('*')).alias('{0}')".format(attr)
                for attr in self.attributes]
            pre_code.extend([
                "# Computes the ratio of missing values for each attribute",
                "ratio_{0} = {0}.select({1}).collect()".format(
                    self.inputs[0], ', '.join(select_list)), "",
                "attributes_{0} = [c for c in {1} "
                "\n        if {2} <= ratio_{0}[0][c] <= {3}]".format(
                    self.inputs[0], attrs_json, self.min_missing_ratio,
                    self.max_missing_ratio)
            ])
        else:
            pre_code.append(
                "attributes_{0} = {1}".format(self.inputs[0], attrs_json))

        if self.cleaning_mode == self.REMOVE_ROW:
            partial.append("""
                {0} = {1}.na.drop(how='any', subset=attributes_{1})""".format(
                output, self.inputs[0]))

        elif self.cleaning_mode == self.VALUE:
            value = ast.literal_eval(self.value)
            if not (isinstance(value, int) or isinstance(value, float)):
                value = '"{}"'.format(value)
            partial.append(
                "\n    {0} = {1}.na.fill(value={2}, "
                "subset=attributes_{1})".format(output, self.inputs[0], value))

        elif self.cleaning_mode == self.REMOVE_COLUMN:
            # Based on http://stackoverflow.com/a/35674589/1646932"
            partial.append(
                "\n{0} = {1}.select("
                "[c for c in {1}.columns if c not in attributes_{1}])".format(
                    output, self.inputs[0]))

        elif self.cleaning_mode == self.MODE:
            # Based on http://stackoverflow.com/a/36695251/1646932
            partial.append("""
                md_replace_{1} = dict()
                for md_attr_{1} in attributes_{1}:
                    md_count_{1} = {0}.groupBy(md_attr_{1}).count()\\
                        .orderBy(desc('count')).limit(1)
                    md_replace_{1}[md_attr_{1}] = md_count_{1}.collect()[0][0]
             {0} = {1}.fillna(value=md_replace_{1})""".format(
                output, self.inputs[0])
            )

        elif self.cleaning_mode == self.MEDIAN:
            # See http://stackoverflow.com/a/31437177/1646932
            # But null values cause exception, so it needs to remove them
            partial.append("""
                mdn_replace_{1} = dict()
                for mdn_attr_{1} in attributes_{1}:
                    # Computes median value for column with relat. error = 10%
                    mdn_{1} = {1}.na.drop(subset=[mdn_attr_{1}])\\
                        .approxQuantile(mdn_attr_{1}, [.5], .1)
                    md_replace_{1}[mdn_attr_{1}] = mdn_{1}[0]
                {0} = {1}.fillna(value=mdn_replace_{1})""".format(
                output, self.inputs[0]))

        elif self.cleaning_mode == self.MEAN:
            partial.append("""
                avg_{1} = {1}.select([avg(c).alias(c)
                                        for c in attributes_{1}]).collect()
                values_{1} = dict([(c, avg_{1}[0][c]) for c in attributes_{1}])
                {0} = {1}.na.fill(value=values_{1})""".format(output,
                                                              self.inputs[0]))
        else:
            raise ValueError(
                "Parameter '{}' has an incorrect value '{}' in {}".format(
                    self.CLEANING_MODE_PARAM, self.cleaning_mode,
                    self.__class__))

        return '\n'.join(pre_code) + \
               "\nif len(attributes_{0}) > 0:".format(self.inputs[0]) + \
               '\n    '.join([dedent(line) for line in partial]).replace(
                   '\n',
                   '\n    ') + \
               "\nelse:\n    {0} = {1}".format(output, self.inputs[0])


class AddColumns(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    Implementation based on post http://stackoverflow.com/a/40510320/1646932
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(inputs) == 2

    def generate_code(self):
        if self.has_code:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = """
                w_{0}_{1} = Window().orderBy()
                {0}_inx =  {0}.withColumn("_inx", rowNumber().over(w_{0}_{1}))
                {1}_inx =  {1}.withColumn("_inx", rowNumber().over(w_{0}_{1})
                {2} = {0}_indexed.join({1}_inx, {0}_inx._inx == {0}_inx._inx,
                                             'inner')
                    .drop({0}_inx._inx).drop({1}_inx._inx)
                """.format(self.inputs[0], self.inputs[1], output)
            return dedent(code)

        return ""


class Replace(Operation):
    """
    Replaces values of columns by specified value. Similar to Transformation
    operation.
    @deprecated
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

    def generate_code(self):
        if self.has_code:
            output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
                self.inputs[0])
            code = output

            return dedent(code)

        return ""
