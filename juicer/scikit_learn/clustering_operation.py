# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation
from juicer.operation import ReportOperation
from juicer.scikit_learn.util import get_X_train_data
from juicer.scikit_learn.model_operation import AlgorithmOperation
from gettext import gettext


class ClusteringModelOperation(Operation):
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'prediction'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) >= 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        if self.has_code:
            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.model = self.named_outputs.get(
                    'model', 'model_task_{}'.format(self.order))

            self.output = self.named_outputs.get(
                    'output data', 'out_task_{}'.format(self.order))
            self.alias = parameters.get(self.ALIAS_PARAM, 'prediction')

            self.transpiler_utils.add_custom_function('get_X_train_data',
                                                      get_X_train_data)
            self.metrics_code = ""

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            """Generate code."""

            copy_code = ".copy()" \
                if self.parameters['multiplicity'].get(
                    'train input data', 0) > 1 or \
                self.parameters['multiplicity'].get(
                           'input data', 0) > 1\
                else ""

            code = """
        X = get_X_train_data({input}, {features})
        clustering_model = {algorithm}.fit(X)

        if hasattr(clustering_model, 'predict'):
            y = clustering_model.predict(X)
        elif hasattr(clustering_model, 'labels_'):
            y = clustering_model.labels_
        else:
            y = clustering_model.transform(X).tolist()

        {OUT} = {input}{copy_code}
        {OUT}['{predCol}'] = y
        {model} = clustering_model
        display_text = {display_text}
        if display_text:
            metric_rows = [{metrics_append}]

            if metric_rows:
                metrics_content = SimpleTableReport(
                    'table table-striped table-bordered w-auto', [],
                    metric_rows,
                    title='{metrics}')

                emit_event('update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=metrics_content.generate(),
                    type='HTML', title='{metrics}',
                    task={{'id': '{task_id}' }},
                    operation={{'id': {operation_id} }},
                    operation_id={operation_id})
            """.format(model=self.model, features=self.features,
                       input=self.named_inputs['train input data'],
                       algorithm=self.named_inputs['algorithm'],
                       copy_code=copy_code, OUT=self.output,
                       predCol=self.alias,
                       metrics_append=self.metrics_code,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       title=_("Clustering result"),
                       summary=gettext('Summary'),
                       metrics=gettext('Metrics'),
                       weights=gettext('Weights'),
                       compute_cost=gettext('Compute cost'),

                       msg1=_('Regression only support numerical features.'),
                       msg2=_('Features are not assembled as a vector. '
                              'They will be implicitly assembled and rows with '
                              'null values will be discarded. If this is '
                              'undesirable, explicitly add a feature assembler '
                              'in the workflow.'),
                       display_text=self.parameters['task']['forms'].get(
                               'display_text', {}).get('value') in (1, '1')
                       )

            return dedent(code)


class ClusteringOperation(AlgorithmOperation):
    def __init__(self, parameters, named_inputs, named_outputs, algorithm):
        input_data = named_inputs.get('train input data')
        if input_data is None:
            input_data = named_inputs.get('input data')
        model_in_ports = {
            'train input data': input_data,
            'algorithm': 'algorithm'}
        model = ClusteringModelOperation(
            parameters, model_in_ports, named_outputs)
        super(ClusteringOperation, self).__init__(
            parameters, named_inputs, named_outputs, model, algorithm)
        model.metrics_code = algorithm.get_output_metrics_code()


class KMeansModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = KMeansClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(KMeansModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class AgglomerativeModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = AgglomerativeClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        # translation agglomerative attributes to clustermodel attributes
        if 'attributes' in parameters:
            parameters['features'] = parameters['attributes']
        if 'alias' in parameters:
            parameters['prediction'] = parameters['alias']
        super(AgglomerativeModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class DBSCANClusteringModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = DBSCANClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(DBSCANClusteringModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class GaussianMixtureClusteringModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = GaussianMixtureClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(GaussianMixtureClusteringModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class LdaClusteringModelOperation(ClusteringOperation):
    def __init__(self, parameters, named_inputs, named_outputs):
        algorithm = LdaClusteringOperation(
                parameters, named_inputs, {'algorithm': 'algorithm'})
        super(LdaClusteringModelOperation, self).__init__(
                parameters, named_inputs, named_outputs, algorithm)


class AgglomerativeClusteringOperation(Operation):
    FEATURES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    N_CLUSTERS_PARAM = 'number_of_clusters'
    LINKAGE_PARAM = 'linkage'
    AFFINITY_PARAM = 'affinity'

    AFFINITY_PARAM_EUCL = 'euclidean'
    AFFINITY_PARAM_L1 = 'l1'
    AFFINITY_PARAM_L2 = 'l2'
    AFFINITY_PARAM_MA = 'manhattan'
    AFFINITY_PARAM_COS = 'cosine'
    AFFINITY_PARAM_PRE = 'precomputed'

    LINKAGE_PARAM_WARD = 'ward'
    LINKAGE_PARAM_COMP = 'complete'
    LINKAGE_PARAM_AVG = 'average'
    LINKAGE_PARAM_SIN = 'single'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.FEATURES_PARAM not in parameters:
            raise ValueError(
                     _("Parameter '{}' must be informed for task {}").format(
                             self.FEATURES_PARAM, self.__class__))

        self.n_clusters = int(parameters.get(self.N_CLUSTERS_PARAM, 2) or 2)
        self.linkage = parameters.get(
            self.LINKAGE_PARAM,
            self.LINKAGE_PARAM_WARD) or self.LINKAGE_PARAM_WARD
        self.affinity = parameters.get(
            self.AFFINITY_PARAM,
            self.AFFINITY_PARAM_EUCL) or self.AFFINITY_PARAM_EUCL

        if self.n_clusters <= 0:
            raise ValueError(
                     _("Parameter '{}' must be x>0 for task {}").format(
                            self.N_CLUSTERS_PARAM, self.__class__))

        self.transpiler_utils.add_import(
                "from sklearn.cluster import AgglomerativeClustering")
        self.transpiler_utils. \
            add_import("from sklearn.metrics import silhouette_score")
        self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)

    @staticmethod
    def get_output_metrics_code():
        code = """
            ['{silhouette_euclidean}', 
                silhouette_score(X, y, metric='euclidean')],
            ['{silhouette_cosine}', 
                silhouette_score(X, y, metric='cosine')],
            ['{n_clusters}', clustering_model.n_clusters_],
            ['{n_leaves}', clustering_model.n_leaves_],
            ['{n_connected_components}', 
                clustering_model.n_connected_components_]
            """.format(silhouette_euclidean=
                       gettext('Silhouette (Euclidean distance)'),
                       silhouette_cosine=
                       gettext('Silhouette (Cosine distance)'),
                       n_clusters=gettext('Number of clusters'),
                       n_leaves=gettext('Number of leaves'),
                       n_connected_components=
                       gettext('Number of connected components'))
        return code

    def generate_code(self):
        code = """
        algorithm = AgglomerativeClustering(n_clusters={n_clusters}, 
            linkage='{linkage}', affinity='{affinity}')
        """.format(n_clusters=self.n_clusters,
                   affinity=self.affinity,
                   linkage=self.linkage)
        return dedent(code)


class DBSCANClusteringOperation(Operation):
    EPS_PARAM = 'eps'
    MIN_SAMPLES_PARAM = 'min_samples'
    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'
    METRIC_PARAM = 'metric'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.eps = float(parameters.get(self.EPS_PARAM, 0.5) or 0.5)
            self.min_samples = int(parameters.get(self.MIN_SAMPLES_PARAM, 5)
                                   or 5)
            self.metric = parameters.get(self.METRIC_PARAM, 'euclidean')

            vals = [self.eps, self.min_samples]
            atts = [self.EPS_PARAM, self.MIN_SAMPLES_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.cluster import DBSCAN")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils. \
                add_import("from sklearn.metrics import silhouette_score")

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{silhouette_euclidean}', silhouette_score(X, y, metric='euclidean')],
        ['{silhouette_cosine}', silhouette_score(X, y, metric='cosine')]
        """.format(silhouette_euclidean=
                   gettext('Silhouette (Euclidean distance)'),
                   silhouette_cosine=
                   gettext('Silhouette (Cosine distance)'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = DBSCAN(eps={eps}, min_samples={min_samples}, 
        metric='{metric}')
        """.format(eps=self.eps, min_samples=self.min_samples,
                   metric=self.metric)
        return dedent(code)


class GaussianMixtureClusteringOperation(Operation):
    MAX_ITER_PARAM = 'max_iter'
    TOL_PARAM = 'tol'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'
    N_COMPONENTS_PARAM = 'n_components'
    COVARIANCE_TYPE_PARAM = 'covariance_type'
    REG_COVAR_PARAM = 'reg_covar'
    N_INIT_PARAM = 'n_init'
    RANDOM_STATE_PARAM = 'random_state'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if self.has_code:
            self.n_components = int(parameters.get(self.N_COMPONENTS_PARAM, 1)
                                    or 1)
            self.covariance_type = parameters.get(self.COVARIANCE_TYPE_PARAM,
                                                  'full')
            self.tol = float(parameters.get(self.TOL_PARAM, 0.001) or 0.001)
            self.tol = abs(self.tol)
            self.reg_covar = float(parameters.get(self.REG_COVAR_PARAM,
                                                  0.000001))
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.n_init = int(parameters.get(self.N_INIT_PARAM, 1))
            self.random_state = parameters.get(self.RANDOM_STATE_PARAM, None)

            vals = [self.n_components, self.max_iter]
            atts = [self.N_COMPONENTS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                "from sklearn.mixture import GaussianMixture")
            self.transpiler_utils. \
                add_import("from sklearn.metrics import silhouette_score")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.input_treatment()

    def input_treatment(self):
        if self.random_state is not None and self.random_state != '0':
            self.random_state = int(self.random_state)
        else:
            self.random_state = None

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{silhouette_euclidean}', silhouette_score(X, y, metric='euclidean')],
        ['{silhouette_cosine}', silhouette_score(X, y, metric='cosine')],
        ['{weights}', clustering_model.weights_],
        ['{means}', clustering_model.means_],
        ['{converged}', clustering_model.converged_]
        """.format(weights=gettext('Weights'),
                   means=gettext('Mean of each mixture component'),
                   converged=gettext('Convergence'),
                   silhouette_euclidean=
                   gettext('Silhouette (Euclidean distance)'),
                   silhouette_cosine=
                   gettext('Silhouette (Cosine distance)'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = GaussianMixture(n_components={k}, max_iter={iter}, 
            tol={tol}, covariance_type='{covariance_type}', 
            reg_covar={reg_covar}, n_init={n_init}, 
            random_state={random_state})
        """.format(k=self.n_components, iter=self.max_iter,
                   tol=self.tol, covariance_type=self.covariance_type,
                   reg_covar=self.reg_covar, n_init=self.n_init,
                   random_state=self.random_state)
        return dedent(code)


class KMeansClusteringOperation(Operation):

    N_CLUSTERS_PARAM = 'n_clusters'
    INIT_PARAM = 'init'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tolerance'
    TYPE_PARAM = 'type'
    SEED_PARAM = 'random_state'
    N_INIT_PARAM = 'n_init'
    N_INIT_MB_PARAM = 'n_init_mb'
    N_JOBS_PARAM = 'n_jobs'
    ALGORITHM_PARAM = 'algorithm'
    BATCH_SIZE_PARAM = 'batch_size'
    TOL_PARAM = 'tol'
    MAX_NO_IMPROVEMENT_PARAM = 'max_no_improvement'
    PREDICTION_PARAM = 'prediction'
    FEATURES_PARAM = 'features'

    INIT_PARAM_RANDOM = 'random'
    INIT_PARAM_KM = 'K-Means++'
    TYPE_PARAM_KMEANS = 'K-Means'
    TYPE_PARAM_MB = 'Mini-Batch K-Means'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.has_code:
            self.n_init = int(parameters.get(self.N_INIT_PARAM, 10) or 10)
            self.n_jobs = parameters.get(self.N_JOBS_PARAM, None) or None
            self.algorithm = parameters.get(self.ALGORITHM_PARAM, 'auto') or 'auto'
            self.n_init_mb = int(parameters.get(self.N_INIT_MB_PARAM, 3) or 3)
            self.tol = float(parameters.get(self.TOL_PARAM, 0.0) or 0.0)
            self.max_no_improvement = int(parameters.get(
                    self.MAX_NO_IMPROVEMENT_PARAM, 10) or 10)
            self.batch_size = int(parameters.get(self.BATCH_SIZE_PARAM, 100) or 100)
            self.n_clusters = int(parameters.get(self.N_CLUSTERS_PARAM, 8) or 8)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.init_mode = parameters.get(
                    self.INIT_PARAM, self.INIT_PARAM_KM) or self.INIT_PARAM_KM
            self.init_mode = self.init_mode.lower()
            self.tolerance = parameters.get(self.TOLERANCE_PARAM, 1e-4)
            self.tolerance = abs(float(self.tolerance))
            self.seed = parameters.get(self.SEED_PARAM, None) or None
            self.type = parameters.get(
                    self.TYPE_PARAM,
                    self.TYPE_PARAM_KMEANS) or self.TYPE_PARAM_KMEANS

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_CLUSTERS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.type.lower() == "k-means":
                self.transpiler_utils.add_import(
                    "from sklearn.cluster import KMeans")
            else:
                self.transpiler_utils.add_import(
                    "from sklearn.cluster import MiniBatchKMeans")

            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils. \
                add_import("from sklearn.metrics import silhouette_score")
            self.input_treatment()

    def input_treatment(self):
        if self.seed is not None and self.seed != '0':
            self.seed = int(self.seed)
        else:
            self.seed = None

        if self.n_jobs is not None and self.n_jobs != '0':
            self.n_jobs = int(self.n_jobs)
        else:
            self.n_jobs = None

    @staticmethod
    def get_output_metrics_code():
        code = """
        ['{silhouette_euclidean}', silhouette_score(X, y, metric='euclidean')],
        ['{silhouette_cosine}', silhouette_score(X, y, metric='cosine')],
        ['{cluster_centers}', clustering_model.cluster_centers_],
        ['{inertia}', clustering_model.inertia_]
        """.format(inertia=gettext('Inertia'),
                   cluster_centers=gettext('Cluster centers'),
                   silhouette_euclidean=
                   gettext('Silhouette (Euclidean distance)'),
                   silhouette_cosine=
                   gettext('Silhouette (Cosine distance)'))
        return code

    def generate_code(self):
        """Generate code."""
        if self.type.lower() == "k-means":
            code = """
            algorithm = KMeans(n_clusters={k}, init='{init}', 
                               max_iter={max_iter}, tol={tol}, 
                               random_state={seed}, n_init={n_init}, 
                               n_jobs={n_jobs}, algorithm='{algorithm}')
            """.format(k=self.n_clusters, max_iter=self.max_iter,
                       tol=self.tolerance, init=self.init_mode,
                       seed=self.seed, n_init=self.n_init,
                       n_jobs=self.n_jobs, algorithm=self.algorithm)
        else:
            code = """
            algorithm = MiniBatchKMeans(n_clusters={k}, init='{init}', 
                                max_iter={max_iter}, tol={tol}, 
                                random_state={seed}, n_init={n_init}, 
                                max_no_improvement={max_no_improvement}, 
                                batch_size={batch_size})
            """.format(k=self.n_clusters, max_iter=self.max_iter,
                       tol=self.tol, init=self.init_mode,
                       seed=self.seed, n_init=self.n_init_mb,
                       max_no_improvement=self.max_no_improvement,
                       batch_size=self.batch_size)
        return dedent(code)


class LdaClusteringOperation(Operation):
    N_COMPONENTES_PARAM = 'number_of_topics'
    ALPHA_PARAM = 'doc_topic_pior'
    ETA_PARAM = 'topic_word_prior'
    LEARNING_METHOD_PARAM = 'learning_method'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'

    LEARNING_METHOD_ON = 'online'
    LEARNING_METHOD_BA = 'batch'

    FEATURES_PARAM = 'features'
    PREDICTION_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.has_code:
            self.output = named_outputs.get(
                    'output data', 'sampled_data_{}'.format(self.order))
            self.model = named_outputs.get(
                    'model', 'model_{}'.format(self.order))

            self.n_clusters = int(parameters.get(
                    self.N_COMPONENTES_PARAM, 10) or self.N_COMPONENTES_PARAM)
            self.learning_method = parameters.get(
                    self.LEARNING_METHOD_PARAM,
                    self.LEARNING_METHOD_ON) or self.LEARNING_METHOD_ON
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10

            self.doc_topic_pior = \
                parameters.get(self.ALPHA_PARAM, None) or None
            self.topic_word_prior = parameters.get(self.ETA_PARAM,
                                                   None) or None

            self.seed = parameters.get(self.SEED_PARAM, None) or None

            if self.learning_method not in [self.LEARNING_METHOD_ON,
                                            self.LEARNING_METHOD_BA]:
                raise ValueError(
                    _("Invalid optimizer value '{}' for class {}").format(
                        self.learning_method, self.__class__))

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_COMPONENTES_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.decomposition "
                    "import LatentDirichletAllocation")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)

            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')

            if self.FEATURES_PARAM not in self.parameters:
                msg = _("Parameters '{}' must be informed for task {}")
                raise ValueError(msg.format(
                    self.FEATURES_PARAM, self.__class__.__name__))
            self.features = self.parameters.get(self.FEATURES_PARAM)

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    @staticmethod
    def get_output_metrics_code():
        code = """
            ['{bound}', clustering_model.bound_],
            ['{topic_word_prior}', clustering_model.topic_word_prior_],
            ['{doc_topic_prior}', clustering_model.doc_topic_prior_]
            """.format(bound=gettext('Final perplexity score on training set'),
                       doc_topic_prior=
                       gettext('Prior of document topic distribution'),
                       topic_word_prior=
                       gettext('Prior of topic word distribution'))
        return code

    def generate_code(self):
        """Generate code."""
        code = """
        algorithm = LatentDirichletAllocation(n_components={n_components}, 
            doc_topic_prior={doc_topic_prior}, 
            topic_word_prior={topic_word_prior}, 
            learning_method='{learning_method}', 
            max_iter={max_iter}, random_state={seed})
        """.format(n_components=self.n_clusters, max_iter=self.max_iter,
                   doc_topic_prior=self.doc_topic_pior,
                   topic_word_prior=self.topic_word_prior,
                   learning_method=self.learning_method,
                   seed=self.seed)
        return dedent(code)


class TopicReportOperation(ReportOperation):
    """
    Produces a report for topic identification in text
    """
    TERMS_PER_TOPIC_PARAM = 'terms_per_topic'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        ReportOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.terms_per_topic = parameters.get(self.TERMS_PER_TOPIC_PARAM, 10)

        self.has_code = len(named_inputs) == 3 and any(
            [len(self.named_outputs) > 0, self.contains_results()])
        self.output = self.named_outputs.get('topics',
                                             'topics_{}'.format(self.order))

        if not all([named_inputs.get('model'), named_inputs.get('input data'),
                    named_inputs.get('vocabulary')]):
            raise ValueError(
                _('You must inform all input ports for this operation'))

        self.model = self.named_inputs['model']
        self.vocabulary_input = self.named_inputs.get('vocabulary')
        from juicer.scikit_learn.library.topic_report import gen_top_words
        self.transpiler_utils.add_custom_function(
                'gen_top_words', gen_top_words)

    def get_output_names(self, sep=", "):
        return self.output

    def get_data_out_names(self, sep=','):
        return self.output

    def generate_code(self):
        code = dedent("""    
            {output} = gen_top_words({model}, {vocabulary}, {tpt})
        """.format(model=self.model,
                   tpt=self.terms_per_topic,
                   vocabulary=self.vocabulary_input,
                   output=self.output,
                   input=self.named_inputs['input data'],
                   topic_col='topics',
                   terms_col='terms',
                   term_idx='termIndices',
                   terms_weights='termWeights'))
        return code
