# -*- coding: utf-8 -*-
from textwrap import dedent
from juicer.operation import Operation


class ClusteringModelOperation(Operation):
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'prediction'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 2
        if self.has_code:

            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)[0]
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.model = self.named_outputs.get('model',
                                                'model_{}'.format(self.output))

            self.perform_transformation = 'output data' in self.named_outputs
            if not self.perform_transformation:
                self.output = 'task_{}'.format(self.order)
            else:
                self.output = self.named_outputs['output data']
                self.alias = parameters.get(self.ALIAS_PARAM, 'prediction')

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        """Generate code."""
        code = """
        X = {input}['{features}'].values.tolist()
        {model} = {algorithm}.fit(X)
        """.format(model=self.model, features=self.features,
                   input=self.named_inputs['train input data'],
                   algorithm=self.named_inputs['algorithm'])

        if self.perform_transformation:
            code += """
        y = {algorithm}.predict({IN})
        {OUT} = {IN}
        {OUT}['{predCol}'] = y
        """.format(OUT=self.output, model=self.model,
                   IN=self.named_inputs['train input data'],
                   predCol=self.alias, algorithm=self.named_inputs['algorithm'])
        else:
            code += """
        {output} = None
        """.format(output=self.output)

        return dedent(code)


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

    LINKAGE_PARAM_WARD = 'ward'
    LINKAGE_PARAM_COMP = 'complete'
    LINKAGE_PARAM_AVG = 'average'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0 or self.contains_results()
        if self.has_code:
            self.output = named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))

            if self.FEATURES_PARAM not in parameters:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))

            self.features = parameters.get(self.FEATURES_PARAM)[0]
            self.alias = parameters.get(self.ALIAS_PARAM, 'cluster')

            self.n_clusters = parameters.get(self.N_CLUSTERS_PARAM, 2) or 2
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

            self.has_import = \
                "from sklearn.cluster import AgglomerativeClustering\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}.copy()
        
        X = {output}['{features}'].values.tolist()
        clt = AgglomerativeClustering(n_clusters={n_clusters},
            linkage='{linkage}', affinity='{affinity}')
        {output}['{alias}'] = clt.fit_predict(X)
        """.format(input=self.named_inputs['input data'], output=self.output,
                   features=self.features, alias=self.alias,
                   n_clusters=self.n_clusters,
                   affinity=self.affinity, linkage=self.linkage)

        return dedent(code)


class DBSCANClusteringOperation(Operation):
    EPS_PARAM = 'eps'
    MIN_SAMPLES_PARAM = 'min_samples'
    FEATURES_PARAM = 'features'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0 or self.contains_results()
        if self.has_code:
            self.output = named_outputs.get(
                    'output data', 'output_data_{}'.format(self.order))
            self.eps = parameters.get(
                    self.EPS_PARAM, 0.5) or 0.5
            self.min_samples = parameters.get(self.MIN_SAMPLES_PARAM, 5) or 5

            if self.FEATURES_PARAM in parameters:
                self.features = parameters.get(self.FEATURES_PARAM)[0]
            else:
                raise \
                    ValueError(_("Parameter '{}' must be informed for task {}")
                               .format(self.FEATURES_PARAM, self.__class__))
            self.alias = parameters.get(self.ALIAS_PARAM, 'cluster')

            vals = [self.eps, self.min_samples]
            atts = [self.EPS_PARAM, self.MIN_SAMPLES_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.cluster import DBSCAN\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = {input}.copy()
        
        X = {output}['{features}'].values.tolist()
        clt = DBSCAN(eps={eps}, min_samples={min_samples})
        {output}['{alias}'] = clt.fit_predict(X)
        """.format(eps=self.eps, min_samples=self.min_samples,
                   input=self.named_inputs['input data'], output=self.output,
                   features=self.features, alias=self.alias)

        return dedent(code)


class GaussianMixtureClusteringOperation(Operation):
    N_CLUSTERS_PARAM = 'number_of_clusters'
    MAX_ITER_PARAM = 'max_iterations'
    TOLERANCE_PARAM = 'tolerance'

    def __init__(self, parameters, named_inputs,
                 named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = named_outputs.get(
                    'algorithm', 'clustering_algorithm_{}'.format(self.order))
            self.number_of_clusters = parameters.get(
                    self.N_CLUSTERS_PARAM, 1) or 1
            self.max_iterations = parameters.get(self.MAX_ITER_PARAM,
                                                 100) or 100
            self.tolerance = parameters.get(self.TOLERANCE_PARAM,
                                            0.001) or 0.001
            self.tolerance = abs(float(self.tolerance))

            vals = [self.number_of_clusters, self.max_iterations]
            atts = [self.N_CLUSTERS_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.mixture import GaussianMixture\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = GaussianMixture(n_components={k}, max_iter={iter}, tol={tol})
        """.format(k=self.number_of_clusters,
                   iter=self.max_iterations, tol=self.tolerance,
                   output=self.output)

        return dedent(code)


class KMeansClusteringOperation(Operation):

    N_CLUSTERS_PARAM = 'n_clusters'
    INIT_PARAM = 'init'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tolerance'
    TYPE_PARAM = 'type'
    SEED_PARAM = 'seed'

    INIT_PARAM_RANDOM = 'random'
    INIT_PARAM_KM = 'K-Means++'
    TYPE_PARAM_KMEANS = 'K-Means'
    TYPE_PARAM_MB = 'Mini-Batch K-Means'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = named_outputs.get(
                    'algorithm', 'clustering_algorithm_{}'.format(self.order))

            self.n_clusters = parameters.get(self.N_CLUSTERS_PARAM, 8) or 8
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 300) or 300
            self.init_mode = parameters.get(
                    self.INIT_PARAM, self.INIT_PARAM_KM) or self.INIT_PARAM_KM
            self.init_mode = self.init_mode.lower()
            self.tolerance = parameters.get(self.TOLERANCE_PARAM, 0.001)
            self.tolerance = abs(float(self.tolerance))
            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'
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
                self.has_import = \
                    "from sklearn.cluster import KMeans\n"
            else:
                self.has_import = \
                    "from sklearn.cluster import MiniBatchKMeans\n"

    def generate_code(self):
        """Generate code."""
        if self.type.lower() == "k-means":
            code = """
            {output} = KMeans(n_clusters={k}, init='{init}',
                max_iter={max_iter}, tol={tol}, random_state={seed})
            """.format(k=self.n_clusters, max_iter=self.max_iter,
                       tol=self.tolerance, init=self.init_mode,
                       output=self.output, seed=self.seed)
        else:
            code = """
            {output} = MiniBatchKMeans(n_clusters={k}, init='{init}',
                max_iter={max_iter}, tol={tol}, random_state={seed})
            """.format(k=self.n_clusters, max_iter=self.max_iter,
                       tol=self.tolerance, init=self.init_mode,
                       output=self.output, seed=self.seed)
        return dedent(code)


class LdaClusteringOperation(Operation):
    N_COMPONENTES_PARAM = 'n_components'
    ALPHA_PARAM = 'doc_topic_pior'
    ETA_PARAM = 'topic_word_prior'
    LEARNING_METHOD_PARAM = 'learning_method'
    MAX_ITER_PARAM = 'max_iter'
    SEED_PARAM = 'seed'

    LEARNING_METHOD_ON = 'online'
    LEARNING_METHOD_BA = 'batch'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_outputs) > 0
        if self.has_code:
            self.output = named_outputs.get(
                    'algorithm', 'clustering_algorithm_{}'.format(self.order))
            self.n_clusters = parameters.get(
                    self.N_COMPONENTES_PARAM, 10) or self.N_COMPONENTES_PARAM
            self.learning_method = parameters.get(
                    self.LEARNING_METHOD_PARAM,
                    self.LEARNING_METHOD_ON) or self.LEARNING_METHOD_ON
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 10) or 10

            self.doc_topic_pior = \
                parameters.get(self.ALPHA_PARAM, 'None') or 'None'
            self.topic_word_prior = parameters.get(self.ETA_PARAM,
                                                   'None') or 'None'

            self.seed = parameters.get(self.SEED_PARAM, 'None') or 'None'

            if self.learning_method not in [self.LEARNING_METHOD_ON,
                                            self.LEARNING_METHOD_BA]:
                raise ValueError(
                    _('Invalid optimizer value {} for class {}').format(
                        self.learning_method, self.__class__))

            vals = [self.n_clusters, self.max_iter]
            atts = [self.N_COMPONENTES_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.has_import = \
                "from sklearn.decomposition import LatentDirichletAllocation\n"

    def generate_code(self):
        """Generate code."""
        code = """
        {output} = LatentDirichletAllocation(n_components={n_components}, 
        doc_topic_prior={doc_topic_prior}, topic_word_prior={topic_word_prior}, 
        learning_method='{learning_method}', max_iter={max_iter})
        """.format(n_components=self.n_clusters, max_iter=self.max_iter,
                   doc_topic_prior=self.doc_topic_pior,
                   topic_word_prior=self.topic_word_prior,
                   learning_method=self.learning_method,
                   output=self.output)

        return dedent(code)