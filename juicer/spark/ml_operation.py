# coding=utf-8
import json
import logging
from itertools import izip_longest
from textwrap import dedent

from juicer.operation import Operation, ReportOperation

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
                indexers = [feature.StringIndexer(inputCol=col, outputCol=alias,
                                handleInvalid='skip')
                                    for col, alias in col_alias.iteritems()]

                # Use Pipeline to process all attributes once
                pipeline = Pipeline(stages=indexers)
                models = dict([(col[0], indexers[i].fit({1})) for i, col in
                                enumerate(col_alias)])
                # labels = [model.labels for model in models.itervalues()]

                # Spark ML 2.0.1 do not deal with null in indexer.
                # See SPARK-11569
                {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

                {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
            """.format(self.attributes, self.inputs[0], self.output,
                       json.dumps(zip(self.attributes, self.alias),
                                  indent=None))
        elif self.type == self.TYPE_VECTOR:
            code = """
                col_alias = dict({3})
                indexers = [feature.VectorIndexer(maxCategories={4},
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
            assembler = feature.VectorAssembler(inputCols={0}, outputCol="{1}")
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
        'areaUnderROC': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'areaUnderPR': (
            'evaluation.BinaryClassificationEvaluator', 'rawPredictionCol'),
        'f1': ('evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedPrecision': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'weightedRecall': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'accuracy': (
            'evaluation.MulticlassClassificationEvaluator', 'predictionCol'),
        'rmse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mse': ('evaluation.RegressionEvaluator', 'predictionCol'),
        'mae': ('evaluation.RegressionEvaluator', 'predictionCol'),
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
    NUM_FOLDS_PARAM = 'folds'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)

        self.has_code = len(self.inputs) == 3
        self.num_folds = parameters.get(self.NUM_FOLDS_PARAM, 3)

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['algorithm'],
                          self.named_inputs['input data'],
                          self.named_inputs['evaluator']])

    def get_output_names(self, sep=", "):
        return sep.join([self.output,
                         'eval_{}'.format(self.output),
                         'best_model_{}'.format(self.output)])

    def get_data_out_names(self, sep=','):
        return ''

    def generate_code(self):
        code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid = {algorithm}

            # if estimator.__class__ == classification.LinearRegression:
            #     param_grid = estimator.maxIter
            # elif estimator.__class__  == classification.:
            #     pass
            # elif estimator.__class__ == classification.DecisionTreeClassifier:
            #     # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
            #     param_grid = (estimator.impurity, ['gini', 'entropy'])
            # elif estimator.__class__ == classification.GBTClassifier:
            #     pass
            # elif estimator.__class__ == classification.RandomForestClassifier:
            #     param_grid = estimator.maxDepth
            for param_name, values in param_grid.iteritems():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = {evaluator}

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            evaluated_data = cv_model.transform({input_data})
            best_model_{output}  = cv_model.bestModel
            metric_result = evaluator.evaluate(evaluated_data)
            {output} = evaluated_data
            """.format(algorithm=self.named_inputs['algorithm'],
                       input_data=self.named_inputs['input data'],
                       evaluator=self.named_inputs['evaluator'],
                       output=self.output,
                       folds=self.num_folds))

        # If there is an output needing the evaluation result, it must be
        # processed here (summarization of data results)
        needs_evaluation = 'evaluation' in self.named_outputs
        if needs_evaluation:
            eval_code = """
            grouped_result = evaluated_data.select(
                    evaluator.getLabelCol(), evaluator.getPredictionCol())\\
                    .groupBy(evaluator.getLabelCol(),
                             evaluator.getPredictionCol()).count().collect()
            eval_{output} = {{
                'metric': {{
                    'name': evaluator.getMetricName(),
                    'value': metric_result
                }},
                'estimator': {{
                    'name': estimator.__class__.__name__,
                    'predictionCol': evaluator.getPredictionCol(),
                    'labelCol': evaluator.getLabelCol()
                }},
                'confusion_matrix': {{
                    'data': json.dumps(grouped_result)
                }},
                'evaluator': evaluator
            }}
            """.format(output=self.output)
            code = '\n'.join([code, dedent(eval_code)])

        return code


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
                self.__class__.__name__))

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
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.outputs) > 0
        self.name = "FIXME"
        if 'paramgrid' not in parameters:
            raise ValueError(
                'Parameter grid must be informed for classifier {}'.format(
                    self.__class__))

        if not all([self.LABEL_PARAM in parameters['paramgrid'],
                    self.FEATURES_PARAM in parameters['paramgrid']]):
            msg = "Parameters '{}' and '{}' must be informed for task {}"
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__))

        self.label = parameters['paramgrid'].get(self.LABEL_PARAM)
        self.attributes = parameters['paramgrid'].get(self.FEATURES_PARAM)

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return self.output

    def generate_code(self):
        param_grid = {
            'featuresCol': self.attributes,
            'labelCol': self.label
        }
        declare = dedent("""
        param_grid = {2}
        # Output result is the classifier and its parameters. Parameters are
        # need in classification model or cross valitor.
        {0} = ({1}(), param_grid)
        """).format(self.output, self.name, json.dumps(param_grid, indent=4))

        code = [declare]
        return "\n".join(code)


class SvmClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.parameters = parameters
        self.has_code = False
        self.name = 'classification.SVM'


class LogisticRegressionClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.parameters = parameters
        self.name = 'classification.LogisticRegression'


class DecisionTreeClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'classification.DecisionTreeClassifier'


class GBTClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs,
                                     named_outputs)
        self.name = 'classification.GBTClassifier'


class NaiveBayesClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'classification.NaiveBayes'


class RandomForestClassifierOperation(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'classification.RandomForestClassifier'


class PerceptronClassifier(ClassifierOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClassifierOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.name = 'classification.MultilayerPerceptronClassificationModel'


class ClassificationReport(ReportOperation):
    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ReportOperation.__init__(self, parameters, inputs, outputs,
                                 named_inputs, named_outputs)
        self.has_code = len(self.inputs) > 1
        self.multiple_inputs = True

    def get_data_out_names(self, sep=','):
        return ''

    def generate_code(self):
        code = dedent("""
            {output} = "ok"
        """.format(output=self.output))
        return code




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

        self.output = self.named_outputs['output data']
        self.model = self.named_outputs.get('model', '{}_model'.format(
            self.output))
        # self.named_outputs['output data']))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'], self.model])

    def generate_code(self):
        code = """
        {algorithm}.setFeaturesCol('{features}')
        {model} = {algorithm}.fit({input})
        # There is no way to pass which attribute was used in clustering, so
        # this information will be stored in uid (hack).
        {model}.uid += '|{features}'
        {output} = {model}.transform({input})
        """.format(model=self.model,
                   algorithm=self.named_inputs['algorithm'],
                   input=self.named_inputs['train input data'],
                   output=self.output,
                   features=self.features)

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
    DOC_CONCENTRATION_PARAM = 'doc_concentration'
    TOPIC_CONCENTRATION_PARAM = 'topic_concentration'

    ONLINE_OPTIMIZER = 'online'
    EM_OPTIMIZER = 'em'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.number_of_clusters = int(parameters.get(
            self.NUMBER_OF_TOPICS_PARAM, 10))
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM,
                                        self.ONLINE_OPTIMIZER)
        if self.optimizer not in [self.ONLINE_OPTIMIZER, self.EM_OPTIMIZER]:
            raise ValueError(
                'Invalid optimizer value {} for class {}'.format(
                    self.optimizer, self.__class__))

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)

        self.doc_concentration = self.number_of_clusters * [
            float(parameters.get(self.DOC_CONCENTRATION_PARAM,
                                 self.number_of_clusters)) / 50.0]

        self.topic_concentration = float(
            parameters.get(self.TOPIC_CONCENTRATION_PARAM, 0.1))

        self.set_values = [
            ['DocConcentration', self.doc_concentration],
            ['K', self.number_of_clusters],
            ['MaxIter', self.max_iterations],
            ['Optimizer', "'{}'".format(self.optimizer)],
            ['TopicConcentration', self.topic_concentration],
        ]
        self.has_code = len(self.output) > 1
        self.name = "clustering.LDA"


class KMeansClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_topics'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TYPE_PARAMETER = 'type'
    INIT_MODE_PARAMETER = 'init_mode'
    TOLERANCE_PARAMETER = 'tolerance'

    TYPE_TRADITIONAL = 'kmeans'
    TYPE_BISECTING = 'bisecting'

    INIT_MODE_KMEANS_PARALLEL = 'k-means||'
    INIT_MODE_RANDOM = 'random'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM,
                                                 10)

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.type = parameters.get(self.TYPE_PARAMETER)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        self.set_values = [
            ['MaxIter', self.max_iterations],
            ['K', self.number_of_clusters],
            ['Tol', self.tolerance],
        ]
        if self.type == self.TYPE_BISECTING:
            self.name = "BisectingKMeans"
        elif self.type == self.TYPE_TRADITIONAL:
            if parameters.get(
                    self.INIT_MODE_PARAMETER) == self.INIT_MODE_RANDOM:
                self.init_mode = self.INIT_MODE_RANDOM
            else:
                self.init_mode = self.INIT_MODE_KMEANS_PARALLEL
            self.set_values.append(['InitMode', '"{}"'.format(self.init_mode)])
            self.name = "clustering.KMeans"
        else:
            raise ValueError(
                'Invalid type {} for class {}'.format(
                    self.type, self.__class__))

        self.has_code = len(self.output) > 1


class GaussianMixtureClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_topics'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TOLERANCE_PARAMETER = 'tolerance'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, inputs, outputs,
                                     named_inputs, named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM,
                                                 10)

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        self.set_values = [
            ['MaxIter', self.max_iterations],
            ['K', self.number_of_clusters],
            ['Tol', self.tolerance],
        ]
        self.name = "clustering.GaussianMixture"
        self.has_code = len(self.output) > 1


class TopicReportOperation(ReportOperation):
    """
    Produces a report for topic identification in text
    """
    TERMS_PER_TOPIC_PARAM = 'terms_per_topic'

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        ReportOperation.__init__(self, parameters, inputs, outputs,
                                 named_inputs, named_outputs)
        self.terms_per_topic = parameters.get(self.TERMS_PER_TOPIC_PARAM, 20)

        self.has_code = len(self.inputs) == 3

    def generate_code(self):
        code = dedent("""
            topic_df = {model}.describeTopics(maxTermsPerTopic={tpt})
            # See hack in ClusteringModelOperation
            features = {model}.uid.split('|')[1]
            '''
            for row in topic_df.collect():
                topic_number = row[0]
                topic_terms  = row[1]
                print "Topic: ", topic_number
                print '========================='
                print '\\t',
                for inx in topic_terms[:{tpt}]:
                    print {vocabulary}[features][inx],
                print
            '''
            {output} =  {input}
        """.format(model=self.named_inputs['model'],
                   tpt=self.terms_per_topic,
                   vocabulary=self.named_inputs['vocabulary'],
                   output=self.get_output_names('output data'),
                   input=self.named_inputs['input data']))
        return code


"""
  Collaborative Filtering part
"""


class CollaborativeOperation(Operation):
    """
    Base class for Collaborative Filtering algorithm
    """

    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, inputs, outputs, named_inputs,
                           named_outputs)
        self.has_code = len(self.outputs) > 0
        self.name = "FIXME"
        self.set_values = []

        # Define outputs and model
        self.output = self.named_outputs['output data']
        self.model = self.named_outputs.get('model', '{}_model'.format(
            self.output))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.named_outputs['output data'], self.model])

    def generate_code(self):
        declare = "{0} = {1}()".format(self.output, self.name)
        code = [declare]
        code.extend(['{0}.set{1}({2})'.format(self.output, name, v)
                     for name, v in self.set_values])
        return "\n".join(code)




class AlternatingLeastSquaresOperation(CollaborativeOperation):
    """
        Alternating Least Squares (ALS) matrix factorization.

        The spark algorithm used is based on
        `"Collaborative Filtering for Implicit Feedback Datasets",
        <http://dx.doi.org/10.1109/ICDM.2008.22>`_
    """
    RANK_PARAM = 'rank'
    MAX_ITER_PARAM = 'maxIter'
    USER_COL_PARAM = 'userCol'
    ITEM_COL_PARAM = 'itemCol'
    RATING_COL_PARAM = 'ratingCol'
    REG_PARAM = 'regParam'

    IMPLICIT_PREFS_PARAM = 'implicitPrefs'
    ALPHA_PARAM = 'alpha'
    SEED_PARAM = 'seed'
    NUM_USER_BLOCKS_PARAM = 'numUserBlocks'
    NUM_ITEM_BLOCKS_PARAM = 'numItemBlocks'



    def __init__(self, parameters, inputs, outputs, named_inputs,
                 named_outputs):
        CollaborativeOperation.__init__(self, parameters, inputs, outputs,
                                 named_inputs, named_outputs)

        self.rank = parameters.get(self.RANK_PARAM, 10)
        self.maxIter = parameters.get(self.MAX_ITER_PARAM, 10)
        self.userCol = parameters.get(self.USER_COL_PARAM, 'userId')
        self.itemCol = parameters.get(self.ITEM_COL_PARAM, 'movieId')
        self.ratingCol = parameters.get(self.RATING_COL_PARAM, 'rating')

        self.regParam = parameters.get(self.REG_PARAM, '0.1')
        self.implicitPrefs = parameters.get(self.IMPLICIT_PREFS_PARAM, 'False')

        self.has_code = False
        self.name = "collaborativefiltering.ALS"


    def generate_code(self):
        code = dedent("""
                # Build the recommendation model using ALS on the training data
                als = ALS(maxIter={maxIter}, regParam={regParam},
                        userCol={userCol}, itemCol={itemCol},
                        ratingCol={ratingCol])

                {model} = als.fit({input})
                predictions = model.transform(test)

                # Evaluate the model not support YET
                # evaluator = RegressionEvaluator(metricName="rmse",
                #                labelCol={ratingCol},
                #                predictionCol="prediction")

                # rmse = evaluator.evaluate(predictions)
                # print("Root-mean-square error = " + str(rmse))
                """.format(
                        output=self.named_outputs[0],
                        input=self.named_inputs[0],
                        model=self.model,
                        maxIter=self.maxIter,
                        regParam=self.regParam,
                        userCol=self.userCol,
                        itemCol=self.itemCol,
                        ratingCol=self.ratingCol)
                    )

        return code