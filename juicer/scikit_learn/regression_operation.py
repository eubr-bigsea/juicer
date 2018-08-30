from textwrap import dedent
from juicer.operation import Operation
from itertools import izip_longest


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


class RegressionModelOperation(Operation):
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

        #TODO:  np.ravel() if isotonic

        if self.has_code:
            code = """
            algorithm = {algorithm}
            {output_data} = {input}
            X_train = {input}['{features}'].values.tolist()
            y = {input}['{label}'].values.tolist()
            {model} = algorithm.fit(X_train, y)
            {output_data}['{prediction}'] = algorithm.predict(X_train).tolist()
            """.format(model=self.model, algorithm=self.algorithm,
                       input=self.named_inputs['train input data'],
                       output_data=self.output, prediction=self.prediction,
                       label=self.label[0], features=self.features[0])

            return dedent(code)


class GradientBoostingRegressorOperation(RegressionOperation):
    LEARNING_RATE_PARAM = 'learning_rate'
    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.GradientBoostingRegressor'
        self.has_code = len(named_outputs) > 0

        if self.has_code:
            self.learning_rate = parameters.get(
                    self.LEARNING_RATE_PARAM, 0.1) or 0.1
            self.n_estimators = parameters.get(
                    self.N_ESTIMATORS_PARAM, 100) or 100
            self.max_depth = parameters.get(
                    self.MAX_DEPTH_PARAM, 3) or 3
            self.min_samples_split = parameters.get(
                    self.MIN_SPLIT_PARAM, 2) or 2
            self.min_samples_leaf = parameters.get(
                    self.MIN_LEAF_PARAM, 1) or 1
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.learning_rate, self.n_estimators,
                    self.min_samples_split, self.min_samples_leaf]
            atts = [self.LEARNING_RATE_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

    def generate_code(self):
        code = dedent("""
        from sklearn.ensemble import GradientBoostingRegressor
        {output} = GradientBoostingRegressor(learning_rate={learning_rate},
        n_estimators={n_estimators}, max_depth={max_depth}, 
        min_samples_split={min_samples_split}, 
        min_samples_leaf={min_samples_leaf}, random_state={seed})""".format(
                output=self.output, learning_rate=self.learning_rate,
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                seed=self.seed, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf))
        return code


class HuberRegressorOperation(RegressionOperation):

    """
    Linear regression model that is robust to outliers.
    """

    EPSILON_PARAM = 'epsilon'
    MAX_ITER_PARAM = 'max_iter'
    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.HuberRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.epsilon = parameters.get(self.EPSILON_PARAM, 1.35) or 1.35
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 100) or 100
            self.alpha = parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.00001) or 0.00001
            self.tol = abs(self.tol)

            vals = [self.max_iter, self.alpha]
            atts = [self.MAX_ITER_PARAM, self.ALPHA_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.epsilon <= 1.0:
                raise ValueError(
                        _("Parameter '{}' must be x>1.0 for task {}").format(
                                self.EPSILON_PARAM, self.__class__))

    def generate_code(self):
        code = dedent("""
            from sklearn.linear_model import HuberRegressor
            {output} = HuberRegressor(epsilon={epsilon},
                max_iter={max_iter}, alpha={alpha},
                tol={tol})
            """).format(output=self.output,
                        epsilon=self.epsilon,
                        alpha=self.alpha,
                        max_iter=self.max_iter,
                        tol=self.tol)

        return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    ISOTONIC_PARAM = 'isotonic'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)

    def generate_code(self):
        code = dedent("""
        from sklearn.isotonic import IsotonicRegression
        {output} = IsotonicRegression(increasing={isotonic})
        """).format(output=self.output, isotonic=self.isotonic)
        return code


class LinearRegressionOperation(RegressionOperation):

    ALPHA_PARAM = 'alpha'
    ELASTIC_NET_PARAM = 'l1_ratio'
    NORMALIZE_PARAM = 'normalize'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.LinearRegression'
        self.has_code = len(named_outputs) > 0

        if self.has_code:
            self.alpha = parameters.get(self.ALPHA_PARAM, 1.0) or 1.0
            self.elastic = parameters.get(self.ELASTIC_NET_PARAM,
                                          0.5) or 0.5
            self.normalize = self.parameters.get(self.NORMALIZE_PARAM,
                                                 True) in (1, '1', 'true', True)
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 1000) or 1000
            self.tol = self.parameters.get(
                    self.TOLERANCE_PARAM, 0.0001) or 0.0001
            self.tol = abs(self.tol)
            self.seed = self.parameters.get(self.SEED_PARAM, 'None')

            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if 0 > self.elastic > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_NET_PARAM, self.__class__))

    def generate_code(self):
        code = dedent("""
        from sklearn.linear_model import ElasticNet
        {output} = ElasticNet(alpha={alpha}, l1_ratio={elastic}, tol={tol},
                              max_iter={max_iter}, random_state={seed},
                              normalize={normalize})""".format(
                output=self.output, max_iter=self.max_iter,
                alpha=self.alpha, elastic=self.elastic,
                seed=self.seed, tol=self.tol, normalize=self.normalize))
        return code


class RandomForestRegressorOperation(RegressionOperation):

    """
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    """

    N_ESTIMATORS_PARAM = 'n_estimators'
    MAX_FEATURES_PARAM = 'max_features'
    MAX_DEPTH_PARAM = 'max_depth'
    MIN_SPLIT_PARAM = 'min_samples_split'
    MIN_LEAF_PARAM = 'min_samples_leaf'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.n_estimators = parameters.get(
                    self.N_ESTIMATORS_PARAM, 10) or 10
            self.max_features = parameters.get(
                    self.MAX_FEATURES_PARAM, 'auto') or 'auto'
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, 3) or 3
            self.min_samples_split = parameters.get(
                    self.MIN_SPLIT_PARAM, 2) or 2
            self.min_samples_leaf = parameters.get(
                    self.MIN_LEAF_PARAM, 1) or 1
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.max_depth, self.n_estimators, self.min_samples_split,
                    self.min_samples_leaf]
            atts = [self.MAX_DEPTH_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

    def generate_code(self):
        code = dedent("""
            from sklearn.ensemble import RandomForestRegressor
            {output} = RandomForestRegressor(n_estimators={n_estimators},
                max_features='{max_features}',
                max_depth={max_depth},
                min_samples_split={min_samples_split},
                min_samples_leaf={min_samples_leaf},
                random_state={seed})
            """).format(output=self.output,
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        seed=self.seed)

        return code


class SGDRegressorOperation(RegressionOperation):

    """
    Linear model fitted by minimizing a regularized empirical loss with
    Stochastic Gradient Descent.
    """

    ALPHA_PARAM = 'alpha'
    ELASTIC_PARAM = 'l1_ratio'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'seed'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.SGDRegressor'
        self.has_code = len(self.named_outputs) > 0

        if self.has_code:
            self.alpha = parameters.get(
                    self.ALPHA_PARAM, 0.0001) or 0.0001
            self.l1_ratio = parameters.get(
                    self.ELASTIC_PARAM, 0.15) or 0.15
            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 1000) or 1000
            self.tol = parameters.get(
                    self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(self.tol)
            self.seed = parameters.get(self.SEED_PARAM, 'None')

            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if 0 > self.l1_ratio > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_PARAM, self.__class__))

    def generate_code(self):
        code = dedent("""
            from sklearn.linear_model import SGDRegressor
            {output} = SGDRegressor(alpha={alpha},
                l1_ratio={l1_ratio}, max_iter={max_iter},
                tol={tol}, random_state={seed})
            """).format(output=self.output,
                        alpha=self.alpha,
                        l1_ratio=self.l1_ratio,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        seed=self.seed)

        return code

