# coding=utf-8
from __future__ import unicode_literals, absolute_import

import logging

from juicer.spark.ml_operation import DeployModelMixin, SvmClassifierOperation, \
    LogisticRegressionClassifierOperation, DecisionTreeClassifierOperation, \
    GBTClassifierOperation, NaiveBayesClassifierOperation, \
    RandomForestClassifierOperation, PerceptronClassifier, OneVsRestClassifier, \
    ClassificationModelOperation

try:
    from itertools import zip_longest as zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from textwrap import dedent

from juicer.operation import Operation

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class ClassifierAndModelOperation(ClassificationModelOperation):
    """
    Base class for classification algorithms that are complete operations (
    apply algorithm and generate model).
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super(ClassifierAndModelOperation, self).__init__(
            parameters, named_inputs, named_outputs)


class SvmModelOperation(SvmClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        SvmClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(SvmClassifierOperation, self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class LogisticRegressionModelOperation(LogisticRegressionClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        LogisticRegressionClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(LogisticRegressionClassifierOperation,
                               self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class DecisionTreeModelOperation(DecisionTreeClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        DecisionTreeClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(DecisionTreeClassifierOperation,
                               self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class GBTModelOperation(GBTClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        GBTClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(GBTClassifierOperation, self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class NaiveBayesModelOperation(NaiveBayesClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        NaiveBayesClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(NaiveBayesClassifierOperation,
                               self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class RandomForestModelOperation(RandomForestClassifierOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        RandomForestClassifierOperation.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(RandomForestClassifierOperation,
                               self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class PerceptronModelOperation(PerceptronClassifier):
    def __init__(self, parameters, named_inputs, named_outputs):
        PerceptronClassifier.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(PerceptronClassifier, self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


class OneVsRestModelOperation(OneVsRestClassifier):
    def __init__(self, parameters, named_inputs, named_outputs):
        OneVsRestClassifier.__init__(
            self, parameters, named_inputs, named_outputs)
        self.classification_model = ClassifierAndModelOperation(
            parameters, named_inputs, named_inputs)

    def generate_code(self):
        algorithm_code = super(OneVsRestClassifier, self).generate_code()
        model_code = self.classification_model.generate_code()
        return "\n".join([algorithm_code, model_code])


"""
Clustering part
"""


class ClusteringModelOperation(Operation):
    FEATURES_ATTRIBUTE_PARAM = 'features'
    PREDICTION_ATTRIBUTE_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.FEATURES_ATTRIBUTE_PARAM not in parameters:
            msg = _("Parameter '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_ATTRIBUTE_PARAM, self.__class__))

        self.features = parameters.get(self.FEATURES_ATTRIBUTE_PARAM)

        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('model',
                                            'model_task_{}'.format(self.order))
        self.prediction = parameters.get(self.PREDICTION_ATTRIBUTE_PARAM,
                                         'prediction')

        self.centroids = self.named_outputs.get(
            'cluster centroids', 'centroids_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([
            self.named_inputs.get('train input data',
                                  'train_task_{}'.format(self.order)),
            self.named_inputs.get('algorithm',
                                  'algo_task_{}'.format(self.order))])

    def get_data_out_names(self, sep=','):
        return sep.join([self.output, self.centroids])

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model, self.centroids])

    def generate_code(self):

        if self.has_code:
            code = """
            emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')
            alg = {algorithm}

            # Clone the algorithm because it can be used more than once
            # and this may cause concurrency problems
            params = dict([(p.name, v) for p, v in
                alg.extractParamMap().items()])
            algorithm_cls = globals()[alg.__class__.__name__]
            algorithm = algorithm_cls()
            algorithm.setParams(**params)
            features = {features}
            requires_pipeline = False

            stages = [] # record pipeline stages
            if len(features) > 1 and not isinstance(
                {input}.schema[str(features[0])].dataType, VectorUDT):
                emit(message='{msg2}')
                for f in features:
                    if not dataframe_util.is_numeric({input}.schema, f):
                        raise ValueError('{msg1}')

                # Remove rows with null (VectorAssembler doesn't support it)
                cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                    for c in features])
                stages.append(SQLTransformer(
                    statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))
                final_features = 'features_tmp'
                stages.append(feature.VectorAssembler(
                    inputCols=features, outputCol=final_features))
                requires_pipeline = True

            else:
                # If more than 1 vector is passed, use only the first
                final_features = features[0]

            algorithm.setFeaturesCol(final_features)

            if hasattr(algorithm, 'setPredictionCol'):
                algorithm.setPredictionCol('{prediction}')
            stages.append(algorithm)
            pipeline = Pipeline(stages=stages)

            pipeline_model = pipeline.fit({input})
            {model} = pipeline_model

            # There is no way to pass which attribute was used in clustering, so
            # information will be stored in a new attribute called features.
            setattr({model}, 'features', {features})

            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                if requires_pipeline:
                    return pipeline_model.transform(df).drop(final_features)
                else:
                    return pipeline_model.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_transform)

            summary = getattr({model}, 'summary', None)

            # Lazy execution in case of sampling the data in UI
            def call_clusters(clustering_model):
                if hasattr(clustering_model, 'clusterCenters'):
                    centers = clustering_model.clusterCenters()
                    df_data = [center.tolist() for center in centers]
                    return spark_session.createDataFrame(
                        df_data, ['centroid_{{}}'.format(i)
                            for i in range(len(df_data[0]))])
                else:
                    return spark_session.createDataFrame([],
                        types.StructType([]))

            # Last stage contains clustering model
            {centroids} = dataframe_util.LazySparkTransformationDataframe(
                {model}.stages[-1], {model}.stages[-1], call_clusters)

            if summary:
                summary_rows = []
                for p in dir(summary):
                    if not p.startswith('_') and p != "cluster":
                        try:
                            summary_rows.append(
                                [p, getattr(summary, p)])
                        except Exception as e:
                            summary_rows.append([p, e.message])
                summary_content = SimpleTableReport(
                    'table table-striped table-bordered', [],
                    summary_rows,
                    title='Summary')
                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=summary_content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
            """.format(model=self.model,
                       algorithm=self.named_inputs['algorithm'],
                       input=self.named_inputs['train input data'],
                       output=self.output,
                       features=repr(self.features),
                       prediction=self.prediction,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       centroids=self.centroids,
                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'), )

            return dedent(code)


class ClusteringOperation(Operation):
    """
    Base class for clustering algorithms
    """

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "BaseClustering"
        self.set_values = []
        self.output = self.named_outputs.get('algorithm',
                                             'algo_task_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=','):
        return self.output

    def generate_code(self):
        if self.has_code:
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

    def __init__(self, parameters, named_inputs, named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = int(parameters.get(
            self.NUMBER_OF_TOPICS_PARAM, 10))
        self.optimizer = parameters.get(self.OPTIMIZER_PARAM,
                                        self.ONLINE_OPTIMIZER)
        if self.optimizer not in [self.ONLINE_OPTIMIZER, self.EM_OPTIMIZER]:
            raise ValueError(
                _('Invalid optimizer value {} for class {}').format(
                    self.optimizer, self.__class__))

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)

        self.doc_concentration = parameters.get(self.DOC_CONCENTRATION_PARAM)
        # if not self.doc_concentration:
        #     self.doc_concentration = self.number_of_clusters * [-1.0]

        self.topic_concentration = parameters.get(
            self.TOPIC_CONCENTRATION_PARAM)

        self.set_values = [
            ['K', self.number_of_clusters],
            ['MaxIter', self.max_iterations],
            ['Optimizer', "'{}'".format(self.optimizer)],
        ]
        if self.doc_concentration:
            try:
                doc_concentration = [float(v) for v in
                                     str(self.doc_concentration).split(',') if
                                     v.strip()]
                self.set_values.append(['DocConcentration', doc_concentration])
            except Exception:
                raise ValueError(
                    _('Invalid document concentration: {}. It must be a single '
                      'decimal value or a list of decimal numbers separated by '
                      'comma.').format(
                        self.doc_concentration))

        if self.topic_concentration:
            self.set_values.append(
                ['TopicConcentration', self.topic_concentration])
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "clustering.LDA"


class KMeansClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_clusters'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TYPE_PARAMETER = 'type'
    INIT_MODE_PARAMETER = 'init_mode'
    TOLERANCE_PARAMETER = 'tolerance'

    TYPE_TRADITIONAL = 'kmeans'
    TYPE_BISECTING = 'bisecting'

    INIT_MODE_KMEANS_PARALLEL = 'k-means||'
    INIT_MODE_RANDOM = 'random'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM,
                                                 10)

        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.type = parameters.get(self.TYPE_PARAMETER)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        if self.type == self.TYPE_BISECTING:
            self.name = "BisectingKMeans"
            self.set_values = [
                ['MaxIter', self.max_iterations],
                ['K', self.number_of_clusters],
            ]
        elif self.type == self.TYPE_TRADITIONAL:
            if parameters.get(
                    self.INIT_MODE_PARAMETER) == self.INIT_MODE_RANDOM:
                self.init_mode = self.INIT_MODE_RANDOM
            else:
                self.init_mode = self.INIT_MODE_KMEANS_PARALLEL
            self.set_values.append(['InitMode', '"{}"'.format(self.init_mode)])
            self.name = "clustering.KMeans"
            self.set_values = [
                ['MaxIter', self.max_iterations],
                ['K', self.number_of_clusters],
                ['Tol', self.tolerance],
                ['InitMode', '"{}"'.format(self.init_mode)]
            ]
        else:
            raise ValueError(
                _('Invalid type {} for class {}').format(
                    self.type, self.__class__))

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])


class GaussianMixtureClusteringOperation(ClusteringOperation):
    K_PARAM = 'number_of_clusters'
    MAX_ITERATIONS_PARAM = 'max_iterations'
    TOLERANCE_PARAMETER = 'tolerance'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ClusteringOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.number_of_clusters = parameters.get(self.K_PARAM, 10)
        self.max_iterations = parameters.get(self.MAX_ITERATIONS_PARAM, 10)
        self.tolerance = float(parameters.get(self.TOLERANCE_PARAMETER, 0.001))

        self.set_values = [
            ['MaxIter', self.max_iterations],
            ['K', self.number_of_clusters],
            ['Tol', self.tolerance],
        ]
        self.name = "clustering.GaussianMixture"
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])


"""
  Collaborative Filtering part
"""


class RecommendationModel(Operation):
    # RANK_PARAM = 'rank'
    # MAX_ITER_PARAM = 'max_iter'
    # USER_COL_PARAM = 'user_col'
    # ITEM_COL_PARAM = 'item_col'
    # RATING_COL_PARAM = 'rating_col'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        # if not all([self.RANK_PARAM in parameters,
        #             self.RATING_COL_PARAM in parameters]):
        #     msg = _("Parameters '{}' and '{}' must be informed for task {}")
        #     raise ValueError(msg.format(
        #         self.RANK_PARAM, self.RATING_COL_PARAM,
        #         self.__class__.__name__))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))
        self.output = self.named_outputs.get(
            'output data', 'data_{}'.format(self.order))
        # self.ratingCol = parameters.get(self.RATING_COL_PARAM)

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        code = """
        {0} = {1}.fit({2})
        {output_data} = {0}.transform({2})
        """.format(self.model, self.named_inputs['algorithm'],
                   self.named_inputs['input data'],
                   output_data=self.output)

        return dedent(code)


class CollaborativeOperation(Operation):
    """
    Base class for Collaborative Filtering algorithm
    """

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "als"
        self.set_values = []
        # Define outputs and model
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('model', 'model_task_{}'.format(
            self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.model])

    def generate_code(self):
        declare = "{0} = {1}()".format(self.output, self.name)
        code = [declare]
        code.extend(['{0}.set{1}({2})'.format(self.output, name, v)
                     for name, v in self.set_values])
        return "\n".join(code)


class AlternatingLeastSquaresOperation(Operation):
    """
        Alternating Least Squares (ALS) matrix factorization.

        The spark algorithm used is based on
        `"Collaborative Filtering for Implicit Feedback Datasets",
        <http://dx.doi.org/10.1109/ICDM.2008.22>`_
    """
    RANK_PARAM = 'rank'
    MAX_ITER_PARAM = 'max_iter'
    USER_COL_PARAM = 'user_col'
    ITEM_COL_PARAM = 'item_col'
    RATING_COL_PARAM = 'rating_col'
    REG_PARAM = 'reg_param'

    IMPLICIT_PREFS_PARAM = 'implicitPrefs'
    ALPHA_PARAM = 'alpha'
    SEED_PARAM = 'seed'
    NUM_USER_BLOCKS_PARAM = 'numUserBlocks'
    NUM_ITEM_BLOCKS_PARAM = 'numItemBlocks'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.rank = parameters.get(self.RANK_PARAM, 10)
        self.maxIter = parameters.get(self.MAX_ITER_PARAM, 10)
        self.userCol = parameters.get(self.USER_COL_PARAM, 'user_id')[0]
        self.itemCol = parameters.get(self.ITEM_COL_PARAM, 'movie_id')[0]
        self.ratingCol = parameters.get(self.RATING_COL_PARAM, 'rating')[0]

        self.regParam = parameters.get(self.REG_PARAM, 0.1)
        self.implicitPrefs = parameters.get(self.IMPLICIT_PREFS_PARAM, False)

        self.has_code = any([len(named_outputs) > 0, self.contains_results()])
        self.name = "collaborativefiltering.ALS"

        # Define input and output
        # output = self.named_outputs['output data']
        # self.input = self.named_inputs['train input data']

    def generate_code(self):
        code = dedent("""
                # Build the recommendation model using ALS on the training data
                # Strategy for dealing with unknown or new users/items at
                # prediction time is drop. See SPARK-14489 and SPARK-19345.
                {algorithm} = ALS(maxIter={maxIter}, regParam={regParam},
                        userCol='{userCol}', itemCol='{itemCol}',
                        ratingCol='{ratingCol}',
                        coldStartStrategy='drop')
                """.format(
            algorithm=self.named_outputs['algorithm'],
            maxIter=self.maxIter,
            regParam=float(self.regParam),
            userCol='{user}'.format(user=self.userCol),
            itemCol='{item}'.format(item=self.itemCol),
            ratingCol='{rating}'.format(rating=self.ratingCol))
        )

        return code


'''
    Logistic Regression Classification
'''


class LogisticRegressionModel(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    WEIGHT_COL_PARAM = ''
    MAX_ITER_PARAM = 'max_iter'
    FAMILY_PARAM = 'family'
    PREDICTION_COL_PARAM = 'prediction'

    REG_PARAM = 'reg_param'
    ELASTIC_NET_PARAM = 'elastic_net'

    # Have summaries model with measure results
    TYPE_BINOMIAL = 'binomial'
    # Multinomial family doesn't have summaries model
    TYPE_MULTINOMIAL = 'multinomial'

    TYPE_AUTO = 'auto'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                           named_outputs)

        self.inputs = None
        self.has_code = any([len(named_outputs) > 0 and len(self.inputs) == 2,
                             self.contains_results()])

        if not all([self.FEATURES_PARAM in parameters['workflow_json'],
                    self.LABEL_PARAM in parameters['workflow_json']]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__.__name__))

        self.model = self.named_outputs.get('model')
        self.output = self.named_outputs.get('output data')

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:

            code = """
            {0} = {1}.fit({2})
            {output_data} = {0}.transform({2})
            """.format(self.model, self.named_inputs['algorithm'],
                       self.named_inputs['input data'], output_data=self.output)

            return dedent(code)
        else:
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format('[]inputs',
                                        '[]outputs',
                                        self.__class__))


'''
    Regression Algorithms
'''


class RegressionModelOperation(DeployModelMixin, Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_COL_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = any([len(named_outputs) > 0 and len(named_inputs) == 2,
                             self.contains_results()])

        if self.has_code:
            self.algorithm = self.named_inputs['algorithm']
            self.input = self.named_inputs['train input data']

            if not all([self.FEATURES_PARAM in parameters,
                        self.LABEL_PARAM in parameters]):
                msg = _("Parameters '{}' and '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.LABEL_PARAM,
                    self.__class__.__name__))

            self.features = parameters[self.FEATURES_PARAM]
            self.label = parameters[self.LABEL_PARAM]
            self.prediction = parameters.get(self.PREDICTION_COL_PARAM,
                                             'prediction') or 'prediction'
            self.model = self.named_outputs.get(
                'model', 'model_{}'.format(self.order))
            self.output = self.named_outputs.get(
                'output data', 'out_task_{}'.format(self.order))

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            code = """
            emit = functools.partial(
                emit_event, name='update task',
                status='RUNNING', type='TEXT',
                identifier='{task_id}',
                operation={{'id': {operation_id}}}, operation_id={operation_id},
                task={{'id': '{task_id}'}},
                title='{title}')
            alg = {algorithm}

            # Clone the algorithm because it can be used more than once
            # and this may cause concurrency problems
            params = dict([(p.name, v) for p, v in
                alg.extractParamMap().items()])

            algorithm_cls = globals()[alg.__class__.__name__]
            algorithm = algorithm_cls()
            algorithm.setParams(**params)
            algorithm.setPredictionCol('{prediction}')
            algorithm.setLabelCol('{label}')

            features = {features}
            requires_pipeline = False

            stages = [] # record pipeline stages
            if len(features) > 1 and not isinstance(
                {input}.schema[str(features[0])].dataType, VectorUDT):
                emit(message='{msg2}')
                for f in features:
                    if not dataframe_util.is_numeric({input}.schema, f):
                        raise ValueError('{msg1}')

                # Remove rows with null (VectorAssembler doesn't support it)
                cond = ' AND '.join(['{{}} IS NOT NULL '.format(c)
                    for c in features])
                stages.append(SQLTransformer(
                    statement='SELECT * FROM __THIS__ WHERE {{}}'.format(cond)))
                final_features = 'features_tmp'
                stages.append(feature.VectorAssembler(
                    inputCols=features, outputCol=final_features))
                requires_pipeline = True

            else:
                # If more than 1 vector is passed, use only the first
                final_features = features[0]

            algorithm.setFeaturesCol(final_features)
            stages.append(algorithm)
            pipeline = Pipeline(stages=stages)

            try:

                pipeline_model = pipeline.fit({input})
                def call_transform(df):
                    if requires_pipeline:
                        return pipeline_model.transform(df).drop(final_features)
                    else:
                        return pipeline_model.transform(df)
                {output_data} = dataframe_util.LazySparkTransformationDataframe(
                    pipeline_model, {input}, call_transform)

                {model} = pipeline_model

                display_text = {display_text}
                if display_text:
                    headers = []
                    row = []
                    metrics = ['coefficients', 'intercept', 'scale', ]

                    for metric in metrics:
                        value = getattr({model}, metric, None)
                        if value:
                            headers.append(metric)
                            row.append(str(value))

                    if row:
                        content = SimpleTableReport(
                            'table table-striped table-bordered',
                            headers, [row])
                        emit_event('update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=content.generate(),
                            type='HTML', title='{title}',
                            task={{'id': '{task_id}' }},
                            operation={{'id': {operation_id} }},
                            operation_id={operation_id})

                    summary = getattr({model}, 'summary', None)
                    if summary:
                        summary_rows = []
                        for p in dir(summary):
                            if not p.startswith('_'):
                                try:
                                    summary_rows.append(
                                        [p, getattr(summary, p)])
                                except Exception as e:
                                    summary_rows.append([p, e.message])
                        summary_content = SimpleTableReport(
                            'table table-striped table-bordered', [],
                            summary_rows,
                            title='Summary')
                        emit_event('update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=summary_content.generate(),
                            type='HTML', title='{title}',
                            task={{'id': '{task_id}' }},
                            operation={{'id': {operation_id} }},
                            operation_id={operation_id})

            except IllegalArgumentException as iae:
                if 'org.apache.spark.ml.linalg.Vector' in iae:
                    raise ValueError('{msg0}')
                else:
                    raise
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label[0], features=repr(self.features),
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title="Regression result",
                       msg0=_('Assemble features in a vector before using a '
                              'regression model'),
                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'),
                       display_text=self.parameters['task']['forms'].get(
                           'display_text', {}).get('value') in (1, '1'))

            return dedent(code)


# noinspection PyAbstractClass
class RegressionOperation(Operation):
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_ATTR_PARAM = 'prediction'

    __slots__ = ('label', 'features', 'prediction')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.label = self.features = self.prediction = None
        self.output = named_outputs.get(
            'algorithm', 'regression_algorithm_{}'.format(self.order))

    def read_common_params(self, parameters):
        if not all([self.LABEL_PARAM in parameters,
                    self.FEATURES_PARAM in parameters]):
            msg = _("Parameters '{}' and '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.FEATURES_PARAM, self.LABEL_PARAM,
                self.__class__))
        else:
            self.label = parameters.get(self.LABEL_PARAM)[0]
            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.prediction = parameters.get(self.PREDICTION_ATTR_PARAM)[0]
            self.output = self.named_outputs.get(
                'algorithm', 'regression_algorithm_{}'.format(self.order))

    def get_output_names(self, sep=', '):
        return self.output

    def get_data_out_names(self, sep=','):
        return ''

    def to_deploy_format(self, id_mapping):
        return []


class LinearRegressionOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    WEIGHT_COL_PARAM = 'weight'
    REG_PARAM = 'reg_param'
    ELASTIC_NET_PARAM = 'elastic_net'

    SOLVER_PARAM = 'solver'

    TYPE_SOLVER_AUTO = 'auto'
    TYPE_SOLVER_NORMAL = 'normal'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.LinearRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            # self.read_common_params(parameters)

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10
            self.reg_param = parameters.get(self.REG_PARAM, 0.1) or 0.0
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)

            self.solver = self.parameters.get(self.SOLVER_PARAM,
                                              self.TYPE_SOLVER_AUTO)
            self.elastic = self.parameters.get(self.ELASTIC_NET_PARAM,
                                               0.0) or 0.0

    def generate_code(self):
        if self.has_code:
            code = dedent("""
            {output} = LinearRegression(
                maxIter={max_iter}, regParam={reg_param},
                solver='{solver}',
                elasticNetParam={elastic})""").format(
                output=self.output,
                max_iter=self.max_iter,
                reg_param=self.reg_param,
                solver=self.solver,
                elastic=self.elastic,
            )
            return code
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class GeneralizedLinearRegressionOperation(RegressionOperation):
    FAMILY_PARAM = 'family'
    LINK_PARAM = 'link'
    MAX_ITER_PARAM = 'max_iter'
    REG_PARAM = 'reg_param'
    WEIGHT_COL_PARAM = 'weight'
    SOLVER_PARAM = 'solver'
    LINK_PREDICTION_COL_PARAM = 'link_prediction'

    TYPE_SOLVER_IRLS = 'irls'
    TYPE_SOLVER_NORMAL = 'normal'

    TYPE_FAMILY_GAUSSIAN = 'gaussian'
    TYPE_FAMILY_BINOMIAL = 'binomial'
    TYPE_FAMILY_POISSON = 'poisson'
    TYPE_FAMILY_GAMMA = 'gamma'

    TYPE_LINK_IDENTITY = 'identity'  # gaussian-poisson-gamma
    TYPE_LINK_LOG = 'log'  # gaussian-poisson-gama
    TYPE_LINK_INVERSE = 'inverse'  # gaussian-gamma
    TYPE_LINK_LOGIT = 'logit'  # binomial-
    TYPE_LINK_PROBIT = 'probit'  # binomial
    TYPE_LINK_CLOGLOG = 'cloglog'  # binomial
    TYPE_LINK_SQRT = 'sqrt'  # poisson

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.GeneralizedLinearRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10)
            self.reg_param = parameters.get(self.REG_PARAM, 0.1)
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)

            self.family = self.parameters.get(self.FAMILY_PARAM,
                                              self.TYPE_FAMILY_BINOMIAL)
            self.link = self.parameters.get(self.LINK_PARAM)

            if self.link is not None:
                self.link = "'{}'".format(self.link)

            # @FIXME: Need to understand the purpose of this parameter
            self.link_prediction_col = self.parameters.get(
                self.LINK_PREDICTION_COL_PARAM, [None])[0]

            if self.link_prediction_col is not None:
                self.link_prediction_col = "'{}'".format(
                    self.link_prediction_col)

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = GeneralizedLinearRegression(
                maxIter={max_iter},
                regParam={reg_param},
                family='{type_family}',
                link={link})
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_iter=self.max_iter,
                        reg_param=self.reg_param,
                        type_family=self.family,
                        link=self.link,
                        link_col=self.link_prediction_col
                        )
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class DecisionTreeRegressionOperation(RegressionOperation):
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INSTANCE_PER_NODE_PARAM = 'min_instance'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    SEED_PARAM = 'seed'

    VARIANCE_COL_PARAM = 'variance_col'
    IMPURITY_PARAM = 'impurity'

    TYPE_IMPURITY_VARIANCE = 'variance'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.DecisionTreeRegressor'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.read_common_params(parameters)

            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5)
            self.min_instance = parameters.get(self.MIN_INSTANCE_PER_NODE_PARAM,
                                               1)
            self.min_info_gain = parameters.get(self.MIN_INFO_GAIN_PARAM, 0.0)

            self.variance_col = self.parameters.get(self.VARIANCE_COL_PARAM,
                                                    None)
            self.seed = self.parameters.get(self.SEED_PARAM, None)

            self.impurity = self.parameters.get(self.IMPURITY_PARAM, 'variance')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = DecisionTreeRegressor(featuresCol='{features}',
                                             labelCol='{label}',
                                             maxDepth={max_iter},
                                             minInstancesPerNode={min_instance},
                                             minInfoGain={min_info},
                                             impurity='{impurity}',
                                             seed={seed},
                                             varianceCol={variance_col}
                                             )
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_depth=self.max_depth,
                        min_instance=self.min_instance,
                        min_info=self.min_info_gain,
                        impurity=self.impurity,
                        seed=self.seed,
                        variance_col=self.variance_col
                        )
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class GBTRegressorOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_INSTANCE_PER_NODE_PARAM = 'min_instance'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    SEED_PARAM = 'seed'

    IMPURITY_PARAM = 'impurity'

    TYPE_IMPURITY_VARIANCE = 'variance'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.GBTRegressor'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5) or 5
            self.min_instance = parameters.get(
                self.MIN_INSTANCE_PER_NODE_PARAM, 1) or 1
            self.min_info_gain = parameters.get(
                self.MIN_INFO_GAIN_PARAM, 0.0) or 0.0

            self.seed = self.parameters.get(self.SEED_PARAM)

            self.impurity = self.parameters.get(self.IMPURITY_PARAM, 'variance')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = GBTRegressor(maxDepth={max_depth},
                                   minInstancesPerNode={min_instance},
                                   minInfoGain={min_info},
                                   seed={seed},
                                   maxIter={max_iter})
            # Only variance is valid, there is a bug, does not work in
            # constructor
            {output}.setImpurity('{impurity}')
            """).format(output=self.output,
                        features=self.features,
                        label=self.label,
                        max_depth=self.max_depth,
                        min_instance=self.min_instance,
                        min_info=self.min_info_gain,
                        impurity=self.impurity,
                        seed=self.seed if self.seed else None,
                        max_iter=self.max_iter)
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class AFTSurvivalRegressionOperation(RegressionOperation):
    MAX_ITER_PARAM = 'max_iter'
    AGR_DETPTH_PARAM = 'aggregation_depth'
    CENSOR_COL_PARAM = 'censor'
    QUANTILES_PROBABILITIES_PARAM = 'quantile_probabilities'
    QUANTILES_COL_PARAM = 'quantiles_col'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.AFTSurvivalRegression'
        self.has_code = any([len(named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 100)
            self.agg_depth = parameters.get(self.AGR_DETPTH_PARAM, 2)

            self.censor = self.parameters.get(
                self.CENSOR_COL_PARAM, ['censor'])[0]
            self.quantile_prob = self.parameters.get(
                self.QUANTILES_PROBABILITIES_PARAM,
                '0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99')
            if self.quantile_prob[0] != '[':
                self.quantile_prob = '[{}]'.format(self.quantile_prob)

            self.quantile_col = self.parameters.get(self.QUANTILES_COL_PARAM,
                                                    'None')

    def generate_code(self):
        if self.has_code:
            declare = dedent("""
            {output} = AFTSurvivalRegression(
                 maxIter={max_iter},
                 censorCol='{censor}',
                 quantileProbabilities={quantile_prob},
                 quantilesCol='{quantile_col}',
                 aggregationDepth={agg_depth}
                 )
            """.format(output=self.output,
                       max_iter=self.max_iter,
                       censor=self.censor,
                       quantile_prob=self.quantile_prob,
                       quantile_col=self.quantile_col,
                       agg_depth=self.agg_depth, ))
            # add , weightCol={weight} if exist
            code = [declare]
            return "\n".join(code)
        else:
            raise ValueError(
                _('Parameter output must be informed for classifier {}').format(
                    self.__class__))


class RandomForestRegressorOperation(RegressionOperation):
    MAX_DEPTH_PARAM = 'max_depth'
    MAX_BINS_PARAM = 'max_bins'
    MIN_INFO_GAIN_PARAM = 'min_info_gain'
    NUM_TREES_PARAM = 'num_trees'
    FEATURE_SUBSET_STRATEGY_PARAM = 'feature_subset_strategy'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 5) or 5
            self.max_bins = parameters.get(self.MAX_BINS_PARAM, 32) or 32
            self.min_info_gain = parameters.get(self.MIN_INFO_GAIN_PARAM,
                                                0.0) or 0.0
            self.num_trees = parameters.get(self.NUM_TREES_PARAM, 20) or 20
            self.feature_subset_strategy = parameters.get(
                self.FEATURE_SUBSET_STRATEGY_PARAM, 'auto')

    def generate_code(self):
        code = dedent("""
            {output} = RandomForestRegressor(
                maxDepth={max_depth},
                maxBins={max_bins},
                minInfoGain={min_info_gain},
                impurity="variance",
                numTrees={num_trees},
                featureSubsetStrategy='{feature_subset_strategy}')
            """).format(output=self.output,
                        max_depth=self.max_depth,
                        max_bins=self.max_bins,
                        min_info_gain=self.min_info_gain,
                        num_trees=self.num_trees,
                        feature_subset_strategy=self.feature_subset_strategy)
        return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    WEIGHT_COL_PARAM = 'weight'
    ISOTONIC_PARAM = 'isotonic'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

        if self.has_code:
            self.weight_col = parameters.get(self.WEIGHT_COL_PARAM, None)
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)

    def generate_code(self):
        code = dedent("""
        {output} = IsotonicRegression(isotonic={isotonic})
        """).format(output=self.output,
                    isotonic=self.isotonic, )
        return code
