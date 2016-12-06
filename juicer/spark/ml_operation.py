# coding=utf-8
import json
import logging
from textwrap import dedent

from juicer.spark.operation import Operation
from itertools import izip_longest

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class ClassifierOperation(Operation):
    """
    Base class for classification algorithms
    """

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

    def generate_code(self):
        pass


class EvaluateModel(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class FeatureIndexer(Operation):
    """
    A label indexer that maps a string attribute of labels to an ML attribute of
    label indices (attribute type = STRING) or a feature transformer that merges
    multiple attributes into a vector attribute (attribute type = VECTOR). All
    other attribute types are first converted to STRING and them indexed.
    """
    ATTRIBUTES_PARAM = 'attributes'
    TYPE_PARAM = 'indexer_type'
    ALIAS_PARAM = 'alias'
    MAX_CATEGORIES_PARAM = 'max_categories'

    TYPE_STRING = 'string'
    TYPE_VECTOR = 'vector'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        '''
        del parameters['workflow_json']
        print '-' * 30
        print parameters.keys()
        print '-' * 30
        '''
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.type = self.parameters.get(self.TYPE_PARAM, self.TYPE_STRING)
        self.alias = [alias.strip() for alias in
                      parameters.get(self.ALIAS_PARAM, '').split(',')]
        self.max_categories = parameters.get(self.MAX_CATEGORIES_PARAM, 20)

        # Adjust alias in order to have the same number of aliases as attributes
        # by filling missing alias with the attribute name sufixed by _indexed.
        self.alias = [x[1] or '{}_indexed'.format(x[0]) for x in
                      izip_longest(self.attributes,
                                   self.alias[:len(self.attributes)])]

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        if self.type == self.TYPE_STRING:
            code = """
                col_alias = dict({3})
                indexers = [StringIndexer(inputCol=col, outputCol=alias,
                                handleInvalid='skip')
                                    for col, alias in col_alias.iteritems()]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=indexers)
                models = dict([(col[0], indexers[i].fit({1})) for i, col in
                                enumerate(col_alias)])
                labels = [model.labels for model in models.itervalues()]

                # Spark ML 2.0.1 do not deal with null in indexer.
                # See SPARK-11569
                {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

                {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
            """.format(self.attributes, self.inputs[0], output,
                       json.dumps(zip(self.attributes, self.alias)))
        elif self.type == self.TYPE_VECTOR:
            code = """
                col_alias = dict({3})
                indexers = [VectorIndexer(maxCategories={4},
                                inputCol=col, outputCol=alias)
                                    for col, alias in col_alias.iteritems()]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=indexers)
                models = dict([(col[0], indexers[i].fit({1})) for i, col in
                                enumerate(col_alias)])
                labels = None

                # Spark ML 2.0.1 do not deal with null in indexer.
                # See SPARK-11569
                {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

                {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
            """.format(self.attributes, self.inputs[0], output,
                       json.dumps(zip(self.attributes, self.alias)),
                       self.max_categories)
        else:
            raise ValueError(
                "Parameter type has an invalid value {}".format(self.type))

        return dedent(code)

    def get_output_names(self, sep=','):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        return sep.join([output, 'models', 'labels'])

    def get_data_out_names(self, sep=','):
        return self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])


class FeatureAssembler(Operation):
    """
    A feature transformer that merges multiple attributes into a vector
    attribute.
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                "Parameter '{}' must be informed for task {}".format(
                    self.ATTRIBUTES_PARAM, self.__class__))
        self.alias = parameters.get(self.ALIAS_PARAM, 'features')

        self.has_code = len(self.inputs) > 0

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])

        code = """
            assembler = VectorAssembler(inputCols={0}, outputCol="{1}")
            {3}_without_null = {3}.na.drop(subset={0})
            {2} = assembler.transform({3}_without_null)
        """.format(json.dumps(self.attributes), self.alias, output,
                   self.inputs[0])

        return dedent(code)


class ApplyModel(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.has_code = False

    def generate_code(self):
        pass


class ClassificationModel(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        self.output = output.replace('df', 'classification')

        self.has_code = len(self.outputs) > 0 and len(self.inputs) == 2
        if self.has_code:
            self.inputs[1] = self.inputs[1].replace('df', 'classifier')

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        return "{0} = ''".format(self.output)


class SvmClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class RandomForestClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, parameters, inputs, outputs)

        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        self.output = output.replace('df', 'classifier')

        self.has_code = len(self.outputs) > 0

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        code = """
        {0} = RandomForestClassifier()
        """.format(self.output)
        return dedent(code)
