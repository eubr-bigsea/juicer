# coding=utf-8
import json
import logging
from textwrap import dedent

from juicer.spark.operation import Operation
from itertools import izip_longest

log = logging.getLogger()
log.setLevel(logging.DEBUG)


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

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
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
            """.format(self.attributes, self.inputs[0], self.output,
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
            """.format(self.attributes, self.inputs[0], self.output,
                       json.dumps(zip(self.attributes, self.alias)),
                       self.max_categories)
        else:
            raise ValueError(
                "Parameter type has an invalid value {}".format(self.type))

        return dedent(code)

    def get_output_names(self, sep=','):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        return sep.join([output, 'models'])

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

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
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
        code = """
            assembler = VectorAssembler(inputCols={0}, outputCol="{1}")
            {3}_without_null = {3}.na.drop(subset={0})
            {2} = assembler.transform({3}_without_null)
        """.format(json.dumps(self.attributes), self.alias, self.output,
                   self.inputs[0])

        return dedent(code)


class ApplyModel(Operation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.inputs) == 2

    def generate_code(self):
        code = """
        {0} = {1}.transform({2})
        """.format(self.output, self.inputs[1], self.inputs[0])

        return dedent(code)


class EvaluateModel(Operation):
    PREDICTION_ATTRIBUTE_PARAM = 'prediction_attribute'
    LABEL_ATTRIBUTE_PARAM = 'label_attribute'
    METRIC_PARAM = 'metric'

    METRIC_TO_EVALUATOR = {
        'areaUnderROC': ('BinaryClassificationEvaluator', 'rawPredictionCol'),
        'areaUnderPR': ('BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedPrecision': (
            'MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedRecall': (
            'MulticlassClassificationEvaluator', 'predictionCol'),
        'accuracy': ('MulticlassClassificationEvaluator', 'predictionCol'),
        'rmse': ('RegressionEvaluator', 'predictionCol'),
        'mse': ('RegressionEvaluator', 'predictionCol'),
        'mae': ('RegressionEvaluator', 'predictionCol'),
    }

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.inputs) == 2
        # @FIXME: validate if metric is compatible with Model using workflow

        self.prediction_attribute = (parameters.get(
            self.PREDICTION_ATTRIBUTE_PARAM) or [''])[0]
        self.label_attribute = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM) or [''])[0]
        self.metric = parameters.get(self.METRIC_PARAM) or ''

        if all([self.prediction_attribute != '', self.label_attribute != '',
                self.metric != '']):
            pass
        else:
            msg = "Parameters '{}', '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format(
                self.PREDICTION_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                self.METRIC_PARAM, self.__class__))
        if self.metric in self.METRIC_TO_EVALUATOR:
            self.evaluator = self.METRIC_TO_EVALUATOR[self.metric][0]
            self.param_prediction_col = self.METRIC_TO_EVALUATOR[self.metric][1]
        else:
            raise ValueError('Invalid metric value {}'.format(self.metric))

        self.has_code = len(self.inputs) > 0 or len(self.output) > 0

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_namesx(self, sep=", "):
        output_evaluator = self.outputs[1] if len(
            self.outputs) > 1 else '{}_tmp_{}'.format(
            self.inputs[0], self.parameters['task']['order'])

        return sep.join([self.output, output_evaluator])

    def generate_code(self):
        code = ''
        if len(self.inputs) > 0:  # Not being used with a cross validator
            code = """
            # Creates the evaluator according to the model
            # (user should not change it)
            evaluator = {6}({7}='{3}',
                                  labelCol='{4}', metricName='{5}')

            {0} = evaluator.evaluate({2})
            """.format(self.output, self.inputs[1], self.inputs[0],
                       self.prediction_attribute, self.label_attribute,
                       self.metric,
                       self.evaluator, self.param_prediction_col)
        elif len(self.output) > 0:  # Used with cross validator
            code = """
            {5} = {0}({1}='{2}',
                            labelCol='{3}', metricName='{4}')
            """.format(self.evaluator, self.param_prediction_col,
                       self.prediction_attribute, self.label_attribute,
                       self.metric, self.output)

        return dedent(code)


class CrossValidationOperation(Operation):
    """
    Cross validation operation used to evaluate classifier results using as many
    as folds provided in input.
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.inputs) == 3

    def get_output_names(self, sep=", "):
        return sep.join([self.output, 'best_model'])

    def get_data_out_names(self, sep=','):
        return ''

    def generate_code(self):
        code = """
            estimator = {0}
            if estimator.__class__ == 'LinearRegression':
                param_grid = estimator.maxIter
            elif estimator.__class__  == NaiveBayes:
                pass
            elif estimator.__class__ == DecisionTreeClassifier:
                # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
                param_grid = (estimator.impurity, ['gini', 'entropy'])
            elif estimator.__class__ == GBTClassifier:
                pass
            elif estimator.__class__ == RandomForestClassifier:
                param_grid = estimator.maxDepth

            evaluator = {2}

            {0}.setLabelCol('survived').setFeaturesCol('features')

            grid = ParamGridBuilder().addGrid(*param_grid).build()
            cv = CrossValidator(estimator=estimator, estimatorParamMaps=grid,
                                evaluator=evaluator)
            cv_model = cv.fit({1})
            best_model  = cv_model.bestModel
            {3} = evaluator.evaluate(cv_model.transform({1}))
            """.format(self.inputs[0], self.inputs[1], self.inputs[2],
                       self.output)
        return dedent(code)


class ClassificationModel(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'
    LABEL_ATTRIBUTE_PARAM = 'label'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.outputs) > 0 and len(self.inputs) == 2
        if not all([self.FEATURES_ATTRIBUTE_PARAM in parameters,
                    self.LABEL_ATTRIBUTE_PARAM in parameters]):
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format(
                self.FEATURES_ATTRIBUTE_PARAM, self.LABEL_ATTRIBUTE_PARAM,
                self.__class__))

        self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM)[0]
        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)[0]

        # @FIXME How to change output name?
        # self.output = output.replace('df', 'classification')
        # if self.has_code:
        #    self.inputs[1] = self.inputs[1].replace('df', 'classifier')

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        code = """
        {1}.setLabelCol('{3}').setFeaturesCol('{4}')
        {0} = {1}.fit({2})
        """.format(self.output, self.inputs[1], self.inputs[0],
                   self.label, self.features)

        return dedent(code)


class ClassifierOperation(Operation):
    """
    Base class for classification algorithms
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.outputs) > 0
        self.name = "FIXME"
        self.set_values = []

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        declare = "{0} = {1}()".format(self.output, self.name)
        code = [declare]
        code.extend([
                        '{0}.set{1}({2})'.format(self.output, name, v)
                        for name, v in self.set_values
                        ])
        return "\n".join(code)


class SvmClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.parameters = parameters
        self.has_code = False
        self.name = 'SVM'


class DecisionTreeClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'DecisionTreeClassifier'


class GBTClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.name = 'GBTClassifier'


class NaiveBayesClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'NaiveBayes'


class RandomForestClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'RandomForestClassifier'
        self.set_values.append(['NumTrees', 10])


"""
Clustering part
"""


class ClusteringModelOperation(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.outputs) > 0 and len(self.inputs) == 2
        if self.FEATURES_ATTRIBUTE_PARAM not in parameters:
            msg = "Parameter '{}' must be informed for task {}"
            raise ValueError(msg.format(
                self.FEATURES_ATTRIBUTE_PARAM, self.__class__))

        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)[0]
        self.model_out = self.named_inputs.get('model', '{}_model_tmp'.format(
            self.inputs[0]))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                         self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model_out])

    def generate_code(self):
        code = """
        {0} = {1}.fit({2})
        {3} =  None
        """.format(self.model_out, self.named_inputs['algorithm'],
                   self.named_inputs['train input data'], self.output)

        return dedent(code)


class ClusteringOperation(Operation):
    """
    Base class for clustering algorithms
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.outputs) > 0
        self.name = "FIXME"
        self.set_values = []

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        declare = "{0} = {1}()".format(self.output, self.name)
        code = [declare]
        code.extend(['{0}.set{1}({2})'.format(self.output, name, v)
                     for name, v in self.set_values])
        return "\n".join(code)


class LdaClusteringOperation(ClusteringOperation):
    NUMBER_OF_TOPICS_PARAM = 'number_of_topics'
    OPTIMIZER_PARAM = 'optimizer'
    MAX_ITERATIONS_PARAM = 'max_iterations'

    ONLINE_OPTIMIZER = 'online'
    EM_OPTIMIZER = 'em'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.number_of_clusters = parameters.get(self.NUMBER_OF_TOPICS_PARAM,
                                                 10)
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM,
                                        self.ONLINE_OPTIMIZER)
        if self.optimizer not in [self.ONLINE_OPTIMIZER, self.EM_OPTIMIZER]:
            raise ValueError(
                'Invalid optimizer value {} for class {}'.format(
                    self.optimizer, self.__class__))

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)

        self.set_values = [
            ['MaxIter', self.max_iterations], ['K', self.number_of_clusters],
            ['Optimizer', "'{}'".format(self.optimizer)],
        ]
        self.has_code = len(self.output) > 1
        self.name = "LDA"


class TopicReportOperation(ClusteringOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.has_code = False

    def generate_code(self):
        pass
