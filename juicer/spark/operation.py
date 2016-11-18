# coding=utf-8
import ast
import json
import logging
from random import random
from textwrap import dedent

import time

from expression import Expression
from metadata import MetadataGet

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class Operation:
    """ Defines an operation in Lemonade """

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        # Indicate if operation generates code or not. Some operations, e.g.
        # Comment, does not generate code
        self.has_code = True

    def test_null_operation(self):
        """
        Test if an operation is null, i.e, does nothing.
        An operation does nothing if it has zero inputs or outputs.
        """
        return any([len(self.outputs) == 0, len(self.inputs) == 0])


class DataReader(Operation):
    """
    Reads a database.
    Parameters:
    - Limonero database ID
    """
    DATA_SOURCE_ID_PARAM = 'data_source'
    HEADER_PARAM = 'header'
    SEPARATOR_PARAM = 'separator'
    INFER_SCHEMA_PARAM = 'infer_schema'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        if self.DATA_SOURCE_ID_PARAM in parameters:
            self.database_id = parameters[self.DATA_SOURCE_ID_PARAM]
            self.header = parameters.get(self.HEADER_PARAM, False)
            self.sep = parameters.get(self.SEPARATOR_PARAM, ',')
            self.infer_schema = parameters.get(self.INFER_SCHEMA_PARAM, True)

            metadata_obj = MetadataGet('123456')
            self.metadata = metadata_obj.get_metadata(self.database_id)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.DATA_SOURCE_ID_PARAM, self.__class__))

    def generate_code(self):

        # For now, just accept CSV files.
        # Should we create a dict with the CSV info at Limonero?
        # such as header and sep.
        code = ''
        if self.metadata['format'] == 'CSV':
            code = """{} = spark.read.csv('{}',
            header={}, sep='{}', inferSchema={})""".format(
                self.outputs[0], self.metadata['url'],
                self.header, self.sep, self.infer_schema)

        elif self.metadata['format'] == 'PARQUET_FILE':
            # TO DO
            pass
        elif self.metadata['format'] == 'JSON_FILE':
            # TO DO
            pass

        return dedent(code)


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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        value = float(parameters.get(self.WEIGHTS_PARAM, 50))
        self.weights = [value, 100 - value]
        self.seed = parameters.get(self.SEED_PARAM, int(random() * time.time()))

    def generate_code(self):
        if len(self.inputs) == 1:
            output1 = self.outputs[0] if len(
                self.outputs) else '{}_tmp1'.format(
                self.inputs[0])
            output2 = self.outputs[1] if len(
                self.outputs) == 2 else '{}_tmp2'.format(
                self.inputs[0])
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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.ascending = map(lambda x: int(x),
                             parameters.get(self.ASCENDING_PARAM,
                                            [1] * len(self.attributes)))
    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        code = "{0} = {1}.orderBy({2}, ascending={3})".format(
            output, self.inputs[0],
            json.dumps(self.attributes),
            json.dumps(self.ascending))

        return dedent(code)


class Distinct(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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


class Sample(Operation):
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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        if self.FRACTION_PARAM in parameters:
            self.withReplacement = parameters.get(self.WITH_REPLACEMENT_PARAM,
                                                  False)
            self.fraction = float(parameters[self.FRACTION_PARAM])
            if not (0 <= self.fraction <= 100):
                msg = "Parameter '{}' must be in range [0, 100] for task {}" \
                    .format(self.FRACTION_PARAM, __name__)
                raise ValueError(msg)
            if self.fraction > 1.0:
                self.fraction *= 0.01

            self.seed = parameters.get(self.SEED_PARAM,
                                       int(random() * time.time()))
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.FRACTION_PARAM, self.__class__))

    def generate_code(self):
        if len(self.outputs) > 0:
            output = self.outputs[0]
        else:
            output = '{}_tmp'.format(self.inputs[0])

        code = "{} = {}.sample(withReplacement={}, fraction={}, seed={})" \
            .format(output, self.inputs[0], self.withReplacement,
                    self.fraction, self.seed)

        return dedent(code)


class Intersection(Operation):
    """
    Returns a new DataFrame containing rows only in both this frame 
    and another frame.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)

    def generate_code(self):
        code = "{} = {}.intersect({})".format(
            self.outputs[0], self.inputs[0], self.inputs[1])
        return dedent(code)


class Difference(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)

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
    JOIN_TYPE_PARAM = 'join_type_param'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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

class ReadCSV(Operation):
    """
    Reads a CSV file without HDFS.
    The purpose of this operation is to read files in
    HDFS without using the Limonero API.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.url = parameters['url']
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"
        try:
            self.separator = parameters['separator']
        except KeyError:
            self.separator = ";"

    def generate_code(self):
        code = """{} = spark.read.csv('{}',
            header={}, sep='{}' ,inferSchema=True)""".format(
            self.outputs[0], self.url, self.header, self.separator)
        return dedent(code)


class Drop(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.alias = parameters['alias']
        self.json_expression = parameters['expression']

    def generate_code(self):
        # Builds the expression and identify the target column
        expression = Expression(self.json_expression)
        built_expression = expression.parsed_expression
        # Builds the code
        code = """{} = {}.withColumn('{}', {})""".format(self.outputs[0],
                                                         self.inputs[0],
                                                         self.alias,
                                                         built_expression)
        return dedent(code)


class Select(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ASCENDING_PARAM = 'ascending'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.group_by = map(lambda x: str(x), parameters['group_by'])
        self.columns = map(lambda x: str(x), parameters['columns'])
        self.function = map(lambda x: str(x), parameters['functions'])
        self.names = map(lambda x: str(x), parameters['new_names'])

        if not all([self.ATTRIBUTES_PARAM in parameters,
                    self.FUNCTION_PARAM in parameters]):
            raise ValueError(
                "Parameters '{}' and {} must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.FUNCTION_PARAM, self.__class__))
        self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        self.function = parameters['function']

    '''
    def generate_code(self):
        elements = []
        for i in range(0, len(self.columns)):
            content = '''{}('{}').alias('{}')'''.format(self.function[i], self.columns[i],
                           self.names[i])
            elements.append(content)
        code = '''{} = {}.groupBy({}).agg({})'''.format(
            self.outputs[0],self.inputs[0], self.group_by, ', '.join(elements))

    '''    
    def generate_code(self):
        info = {self.attributes: self.function}
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        if len(self.inputs) == 1:
            code = """{} = {}.groupBy('{}').agg({})""".format(
                output, self.inputs[0], self.attributes, json.dumps(info))
        else:
            code = ""
        return dedent(code)


class Filter(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.expression = parameters['expression']

    def generate_code(self):
        code = """{} = {}.filter('{}')""".format(
            self.outputs[0], self.inputs[0], self.expression)
        return dedent(code)


<<<<<<< HEAD
class DatetimeToBins(operation):
    '''
    '''
    def __init__(self, parameters, inputs, outputs):
        self.set_io(inputs, outputs)
        self.target_column = parameters['target_column']
        self.new_column = parameters['new_column']
        self.group_size = parameters['group_size']
    def generate_code(self):
        code = '''
            from bins import *
            {} = datetime_to_bins({}, {}, '{}', '{}')
            '''.format(self.outputs[0], self.inputs[0], self.group_size, 
		self.target_column, self.new_column)
        return dedent(code)




class Save(operation):
    """
    Saves the content of the DataFrame at the specified path
    and generate the code to call the Limonero API.
    Parameters:
        - Database name
        - URL for storage
        - Storage ID
        - Database tags
        - Workflow that generated the database
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.name = parameters['name']
        self.format = parameters['format']
        self.url = parameters['url']
        self.storage_id = parameters['storage_id']
        self.tags = ast.literal_eval(parameters['tags'])
        self.workflow = parameters['workflow']
        try:
            self.mode = parameters['mode']
        except KeyError:
            self.mode = "error"
        try:
            self.header = parameters['header']
        except KeyError:
            self.header = "True"

    def generate_code(self):

        code_save = ''
        if self.format == "CSV":
            code_save = """{}.write.csv('{}', header={}, mode='{}')""".format(
                self.inputs[0], self.url, self.header, self.mode)
        elif self.format == "PARQUET":
            pass
        elif self.format == "JSON":
            pass

        code_api = """
            from metadata import MetadataPost

            types_names = dict()
            types_names['IntegerType'] = "INTEGER"
            types_names['StringType'] = "TEXT"
            types_names['LongType'] = "LONG"
            types_names['DoubleType'] = "DOUBLE"
            types_names['TimestampType'] = "DATETIME"


            schema = []
            for att in {0}.schema:
                data = dict()
                data['name'] = att.name
                data['dataType'] = types_names[str(att.dataType)]
                data['nullable'] = att.nullable
                data['metadata'] = att.metadata
                schema.append(data)

            parameters = dict()
            parameters['name'] = "{1}"
            parameters['format'] = "{2}"
            parameters['storage_id'] = {3}
            parameters['provenience'] = str("{4}")
            parameters['description'] = "{5}"
            parameters['user_id'] = "{6}"
            parameters['user_login'] = "{7}"
            parameters['user_name'] = "{8}"
            parameters['workflow_id'] = "{9}"
            parameters['url'] = "{10}"

            instance = MetadataPost('{11}', schema, parameters)
            """.format(self.inputs[0], self.name, self.format, self.storage_id,
                       str(json.dumps(self.workflow)).replace("\"", "'"),
                       self.workflow['workflow']['name'],
                       self.workflow['user']['id'],
                       self.workflow['user']['login'],
                       self.workflow['user']['name'],
                       self.workflow['workflow']['id'], self.url, "123456"
                       )

        code = dedent(code_save) + dedent(code_api)

        return code


class NoOp(Operation):
    """ Null operation """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class SvmClassification(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class EvaluateModel(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


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
        @FIXME: Implement
    """
    ATTRIBUTES_PARAM = 'attributes'
    CLEANING_MODE_PARAM = 'cleaning_mode'
    VALUE_PARAMETER = 'value'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.cleaning_mode = parameters.get(self.CLEANING_MODE_PARAM,
                                            'REMOVE_ROW')

        self.value = parameters.get(self.VALUE_PARAMETER)

        # In this case, nothing will be generated besides create reference to
        # data frame
        if (self.value is None and self.cleaning_mode == 'VALUE') or len(
                self.inputs) == 0:
            self.has_code = False

    def generate_code(self):
        if not self.has_code:
            return "{} = {}".format(self.outputs[0], self.inputs[0])

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        if self.cleaning_mode == 'REMOVE_ROW':
            code = """{} = {}.dropna(how='any', subset=[{}])""".format(
                output, self.inputs[0],
                ', '.join("'{}'".format(x) for x in self.attributes))
        elif self.cleaning_mode == 'VALUE':
            value = ast.literal_eval(self.value)
            if not (isinstance(value, int) or isinstance(value, float)):
                value = '"{}"'.format(value)
            code = """{} = {}.na.fill(value={}, subset=[{}])""".format(
                output, self.inputs[0], value,
                ', '.join("'{}'".format(x) for x in self.attributes))
        elif self.cleaning_mode == 'REMOVE_COLUMN':
            select_list = [
                "(count('{0}') / count('*')).alias('{0}')".format(attr) for
                attr in self.attributes]

            partial = [
                "# Computes which columns have missings and delete them",
                "count_{0} = {0}.select({1}).collect()".format(
                    self.inputs[0], ', '.join(select_list)),
                "drop_{0} = [c for c in {1} if count_{0}[0][c] < 1.0]".format(
                    self.inputs[0], json.dumps(self.attributes)),
                # Based on http://stackoverflow.com/a/35674589/1646932
                "{0} = {1}.select([c for c in {1}.columns if c not in drop_{1}])".format(
                    output,
                    self.inputs[0])

            ]
            code = "\n".join(partial)
        elif self.cleaning_mode == 'MODE':
            code = "@FIXME"
        elif self.cleaning_mode == "MEDIAN":
            # See http://stackoverflow.com/a/31437177/1646932
            # But null values cause exception
            # @FIXME: Not working, need to perform approxQuantile for each attr
            code = """
                # Computes median value for columns",
                mdn_{0} = {0}.dropna().approxQuantile([avg(c).alias(c) for c in {1}]).collect()
                values_{2} = dict([(c, mdn_{0}[0][c]) for c in {1}])
                {2} = {0}.na.fill(value=values_{2})""".format(
                self.inputs[0], json.dumps(self.attributes), output)
            code = "@FIXME"
        elif self.cleaning_mode == 'MEAN':
            code = """
                # Computes mean value for columns",
                avg_{0} = {0}.select([avg(c).alias(c) for c in {1}]).collect()
                values_{2} = dict([(c, avg_{0}[0][c]) for c in {1}])
                {2} = {0}.na.fill(value=values_{2})""".format(
                self.inputs[0], json.dumps(self.attributes), output)
        else:
            code = 'FIXME'
        return dedent(code)


class AddColumns(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    Implementation based on post http://stackoverflow.com/a/40510320/1646932
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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
    Replaces values of columns by specified value
    @FIXME: implementar
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
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
            code = """ """

            return dedent(code)

        return ""
