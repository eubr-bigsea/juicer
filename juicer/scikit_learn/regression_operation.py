# -*- coding: utf-8 -*-

from textwrap import dedent
from juicer.operation import Operation
from itertools import zip_longest
import re
from juicer.scikit_learn.util import get_X_train_data, get_label_data


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
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)

    @property
    def get_inputs_names(self):
        return ', '.join([self.named_inputs['train input data'],
                          self.named_inputs['algorithm']])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        copy_code = ".copy()" \
            if self.parameters['multiplicity']['train input data'] > 1 else ""

        if self.has_code:
            code = """
            algorithm = {algorithm}
            {output_data} = {input}{copy_code}
            X_train = get_X_train_data({input}, {features})
            y = get_label_data({input}, {label})
            if 'IsotonicRegression' in str(algorithm):
                X_train = np.ravel(X_train)
            {model} = algorithm.fit(X_train, y)
            {output_data}['{prediction}'] = algorithm.predict(X_train).tolist()
            """.format(copy_code=copy_code, model=self.model,
                       algorithm=self.algorithm,
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
    MAX_FEATURES_PARAM = 'max_features'
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    RANDOM_STATE_PARAM = 'random_state'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'
    LOSS_PARAM = 'loss'
    SUBSAMPLE_PARAM = 'subsample'
    ALPHA_PARAM = 'alpha'
    CC_APLHA_PARAM = 'cc_alpha'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    TOL_PARAM = 'tol'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.GradientBoostingRegressor'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.learning_rate = float(parameters.get(self.LEARNING_RATE_PARAM, 0.1) or 0.1)
            self.n_estimators = int(parameters.get(self.N_ESTIMATORS_PARAM, 100) or 100)
            self.max_depth = int(parameters.get(self.MAX_DEPTH_PARAM, 3) or 3)
            self.min_samples_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_samples_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.max_features = parameters.get(self.MAX_FEATURES_PARAM, None) or None
            self.criterion = parameters.get(self.CRITERION_PARAM, 'friedman_mse') or 'friedman_mse'
            self.min_weight_fraction_leaf = float(parameters.get(self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0) or 0)
            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM, None)
            self.min_impurity_decrease = float(parameters.get(self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.random_state = parameters.get(self.RANDOM_STATE_PARAM, None)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.subsample = float(parameters.get(self.SUBSAMPLE_PARAM, 1.0) or 1.0)
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.9) or 0.9)
            self.cc_alpha = float(parameters.get(self.CC_APLHA_PARAM, 0) or 0)
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1) or 0.1)
            self.n_iter_no_change = parameters.get(self.N_ITER_NO_CHANGE_PARAM, None) or None
            self.tol = float(parameters.get(self.TOL_PARAM, 1e-4) or 1e-4)
            self.loss = parameters.get(self.LOSS_PARAM, 'ls') or 'ls'

            vals = [self.learning_rate, self.n_estimators,
                    self.min_samples_split, self.min_samples_leaf,
                    self.max_depth]
            atts = [self.LEARNING_RATE_PARAM, self.N_ESTIMATORS_PARAM,
                    self.MIN_SPLIT_PARAM, self.MIN_LEAF_PARAM,
                    self.MAX_DEPTH_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))
            self.transpiler_utils.add_import(
                    "from sklearn.ensemble import GradientBoostingRegressor")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.max_features is not None:
            self.max_features = "'"+self.max_features+"'"
        else:
            self.max_features = 'None'
        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None

        if self.random_state is not None:
            self.random_state = int(self.random_state)
            if self.random_state < 0:
                raise ValueError(
                    _("Parameter '{}' must be x >= 0 or None for task {}").format(
                        self.RANDOM_STATE_PARAM, self.__class__))
        else:
            self.random_state = None

        if self.n_iter_no_change is not None:
            self.n_iter_no_change = int(self.n_iter_no_change)
            if self.n_iter_no_change < 0:
                raise ValueError(
                    _("Parameter '{}' must be x >= 0 or None for task {}").format(
                        self.N_ITER_NO_CHANGE_PARAM, self.__class__))
        else:
            self.n_iter_no_change = None

        if self.cc_alpha < 0:
            raise ValueError(
                _("Parameter '{}' must be x >= 0 for task {}").format(
                    self.CC_APLHA_PARAM, self.__class__))

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x <= 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

        if self.subsample > 1 \
            or self.subsample <= 0:
            raise ValueError(
                _("Parameter '{}' must be 0 < x <= 1 for task {}").format(
                    self.SUBSAMPLE_PARAM, self.__class__))

        if self.min_weight_fraction_leaf > 0.5 \
            or self.min_weight_fraction_leaf < 0:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x <= 0.5 for task {}").format(
                    self.MIN_WEIGHT_FRACTION_LEAF_PARAM, self.__class__))

        if self.min_impurity_decrease < 0:
            raise ValueError(
                _("Parameter '{}' must be x >= 0 for task {}").format(
                    self.MIN_IMPURITY_DECREASE_PARAM, self.__class__))

    def generate_code(self):
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = dedent("""
                {output_data} = {input_data}{copy_code}            
                X_train = get_X_train_data({input_data}, {features})
                y = get_label_data({input_data}, {label})
    
                {model} = GradientBoostingRegressor(loss='{loss}',
                    learning_rate={learning_rate}, 
                    n_estimators={n_estimators}, subsample={subsample}, 
                    criterion='{criterion}', min_samples_split={min_samples_split}, 
                    min_samples_leaf={min_samples_leaf}, 
                    min_weight_fraction_leaf={min_weight_fraction_leaf}, 
                    max_depth={max_depth}, 
                    min_impurity_decrease={min_impurity_decrease}, 
                    random_state={random_state}, max_features={max_features}, 
                    alpha={alpha},
                    max_leaf_nodes={max_leaf_nodes}, 
                    warm_start=False, ccp_alpha={cc_alpha}, 
                    validation_fraction={validation_fraction}, 
                    n_iter_no_change={n_iter_no_change}, tol={tol})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(copy_code=copy_code, output_data=self.output,
                           learning_rate=self.learning_rate,
                           n_estimators=self.n_estimators,
                           max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf,
                           model=self.model,
                           input_data=self.input_port,
                           loss=self.loss,
                           subsample=self.subsample,
                           criterion=self.criterion,
                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                           min_impurity_decrease=self.min_impurity_decrease,
                           random_state=self.random_state,
                           max_features=self.max_features,
                           alpha=self.alpha,
                           max_leaf_nodes=self.max_leaf_nodes,
                           cc_alpha=self.cc_alpha,
                           validation_fraction=self.validation_fraction,
                           n_iter_no_change=self.n_iter_no_change,
                           tol=self.tol,
                           prediction=self.prediction,
                           features=self.features,
                           label=self.label))
            return code


class HuberRegressorOperation(RegressionOperation):

    """
    Linear regression model that is robust to outliers.
    """

    EPSILON_PARAM = 'epsilon'
    MAX_ITER_PARAM = 'max_iter'
    ALPHA_PARAM = 'alpha'
    TOLERANCE_PARAM = 'tol'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.HuberRegressor'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.epsilon = float(parameters.get(self.EPSILON_PARAM, 1.35) or 1.35)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 100) or 100)
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001)
            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.00001) or 0.00001
            self.tol = abs(float(self.tol))
            self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_PARAM, 1) or 1)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')

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

            self.transpiler_utils.add_import(
                    "from sklearn.linear_model import HuberRegressor")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.fit_intercept = True if int(self.fit_intercept) == 1 else False

    def generate_code(self):
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = dedent("""
                {output_data} = {input_data}{copy_code}
                X_train = get_X_train_data({input_data}, {features})
                y = get_label_data({input_data}, {label})
    
                {model} = HuberRegressor(epsilon={epsilon}, max_iter={max_iter}, 
                        alpha={alpha}, tol={tol}, fit_intercept={fit_intercept}, 
                        warm_start=False)
                {model}.fit(X_train, y)
                {output_data}['{predict}'] = {model}.predict(X_train).tolist()
                """).format(copy_code=copy_code,
                            output_data=self.output,
                            epsilon=self.epsilon,
                            alpha=self.alpha,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            input_data=self.input_port,
                            model=self.model,
                            fit_intercept=self.fit_intercept,
                            features=self.features,
                            label=self.label,
                            predict=self.prediction)

            return code


class IsotonicRegressionOperation(RegressionOperation):
    """
        Only univariate (single feature) algorithm supported
    """
    ISOTONIC_PARAM = 'isotonic'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    Y_MIN_PARAM = 'y_min'
    Y_MAX_PARAM = 'y_max'
    OUT_OF_BOUNDS_PARAM = 'out_of_bounds'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.IsotonicRegression'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.isotonic = parameters.get(
                self.ISOTONIC_PARAM, True) in (1, '1', 'true', True)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = parameters.get(self.PREDICTION_PARAM,
                                             'prediction')
            self.y_min = parameters.get(self.Y_MIN_PARAM, None)
            self.y_max = parameters.get(self.Y_MAX_PARAM, None)
            self.out_of_bounds = parameters.get(self.OUT_OF_BOUNDS_PARAM, "nan")

            self.treatment()

            self.transpiler_utils.add_import(
                    "from sklearn.isotonic import IsotonicRegression")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)
            self.treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def treatment(self):
        if len(self.features) >= 2:
            raise ValueError(
                _("Parameter '{}' must be x<2 for task {}").format(
                    self.FEATURES_PARAM, self.__class__))

        if self.y_min is not None and self.y_min != '0':
            self.y_min = float(self.y_min)
        else:
            self.y_min = None

        if self.y_max is not None and self.y_max != '0':
            self.y_max = float(self.y_max)
        else:
            self.y_max = None

        if self.y_max is not None and self.y_min is not None \
            and self.y_min > self.y_max:
            raise ValueError(
                _("Parameter '{}' must be less than or equal to '{}' for "
                  "task {}").format(self.Y_MIN_PARAM, self.Y_MAX_PARAM,
                                    self.__class__))

    def generate_code(self):

        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 \
                else ""

            code = dedent("""
            {model} = IsotonicRegression(y_min={min}, y_max={max}, 
                increasing={isotonic}, out_of_bounds='{bounds}')
    
            {output_data} = {input_data}{copy_code}
            X_train = get_X_train_data({input_data}, {columns})
            X_train = np.ravel(X_train)
            y = get_label_data({input_data}, {label})
        
            {model}.fit(X_train, y)      
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """).format(copy_code=copy_code,
                        output_data=self.output,
                        isotonic=self.isotonic,
                        output=self.output,
                        model=self.model,
                        input_data=self.input_port,
                        min=self.y_min,
                        max=self.y_max,
                        bounds=self.out_of_bounds,
                        columns=self.features,
                        label=self.label,
                        prediction=self.prediction)
            return code


class LinearRegressionOperation(RegressionOperation):

    ALPHA_PARAM = 'alpha'
    ELASTIC_NET_PARAM = 'l1_ratio'
    NORMALIZE_PARAM = 'normalize'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'random_state'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    POSITIVE_PARAM = 'positive'
    FIT_INTERCEPT_PARAM = 'fit_intercept'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.name = 'regression.LinearRegression'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 1.0) or 1.0)
            self.elastic = float(parameters.get(self.ELASTIC_NET_PARAM, 0.5) or 0.5)
            self.normalize = self.parameters.get(self.NORMALIZE_PARAM, True) in (1, '1', 'true', True)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 1000) or 1000)
            self.tol = float(self.parameters.get( self.TOLERANCE_PARAM, 0.0001) or 0.0001)
            seed_ = self.parameters.get(self.SEED_PARAM, None)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM,
                                                  'prediction')

            self.fit_intercept = self.parameters.get(self.FIT_INTERCEPT_PARAM,
                                                     False) == 1
            self.positive = self.parameters.get(self.POSITIVE_PARAM, False) == 1

            vals = [self.alpha, self.max_iter]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            if self.elastic < 0 or self.elastic > 1:
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                self.ELASTIC_NET_PARAM, self.__class__))

            if seed_ is None:
                self.seed = 'None'
            else:
                self.seed = int(seed_)
                if self.seed < 0:
                    raise ValueError(
                        _("Parameter '{}' must be x>=0 for task {}").format(
                                self.SEED_PARAM, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.linear_model import ElasticNet")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def generate_code(self):
        if self.has_code:
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = dedent("""
            {output_data} = {input_data}{copy_code}
            X_train = get_X_train_data({input_data}, {columns})
            y = get_label_data({input_data}, {label})
    
            {model} = ElasticNet(alpha={alpha}, l1_ratio={elastic}, tol={tol}, 
                    max_iter={max_iter}, random_state={seed},
                    normalize={normalize}, positive={positive}, 
                    fit_intercept={fit_intercept})  
            {model}.fit(X_train, y)
            {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
            """.format(copy_code=copy_code,
                       output_data=self.output,
                       max_iter=self.max_iter,
                       alpha=self.alpha,
                       elastic=self.elastic,
                       seed=self.seed,
                       tol=self.tol,
                       normalize=self.normalize,
                       input_data=self.input_port,
                       prediction=self.prediction,
                       columns=self.features,
                       label=self.label,
                       model=self.model,
                       output=self.output,
                       fit_intercept=self.fit_intercept,
                       positive=self.positive))

            return code


class MLPRegressorOperation(Operation):

    HIDDEN_LAYER_SIZES_PARAM = 'layer_sizes'
    ACTIVATION_PARAM = 'activation'
    SOLVER_PARAM = 'solver'
    ALPHA_PARAM = 'alpha'
    MAX_ITER_PARAM = 'max_iter'
    TOLERANCE_PARAM = 'tol'
    SEED_PARAM = 'random_state'

    SOLVER_PARAM_ADAM = 'adam'
    SOLVER_PARAM_LBFGS = 'lbfgs'
    SOLVER_PARAM_SGD = 'sgd'

    ACTIVATION_PARAM_ID = 'identity'
    ACTIVATION_PARAM_LOG = 'logistic'
    ACTIVATION_PARAM_TANH = 'tanh'
    ACTIVATION_PARAM_RELU = 'relu'

    BATCH_SIZE_PARAM = 'batch_size'
    LEARNING_RATE_PARAM = 'learning_rate'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    LEARNING_RATE_INIT_PARAM = 'learning_rate_init'
    POWER_T_PARAM = 'power_t'
    SHUFFLE_PARAM = 'shuffle'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    MOMENTUM_PARAM = 'momentum'
    NESTEROVS_MOMENTUM_PARAM = 'nesterovs_momentum'
    EARLY_STOPPING_PARAM = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    BETA_1_PARAM = 'beta_1'
    BETA_2_PARAM = 'beta_2'
    EPSILON_PARAM = 'epsilon'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.add_functions_required = ""
            self.hidden_layers = str(parameters.get(self.HIDDEN_LAYER_SIZES_PARAM, '(1, 100, 1)'))
            if not re.match(r"(\(\d+, \d+, \d+\))", self.hidden_layers):
                raise ValueError(
                        _("Parameter '{}' must be a tuple with the size "
                          "of each layer for task {}").format(
                                self.HIDDEN_LAYER_SIZES_PARAM, self.__class__))

            self.activation = parameters.get(
                    self.ACTIVATION_PARAM,
                    self.ACTIVATION_PARAM_RELU) or self.ACTIVATION_PARAM_RELU

            self.solver = parameters.get(
                    self.SOLVER_PARAM,
                    self.SOLVER_PARAM_ADAM) or self.SOLVER_PARAM_ADAM

            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001)

            self.max_iter = parameters.get(self.MAX_ITER_PARAM, 200)

            self.tol = float(parameters.get(self.TOLERANCE_PARAM, 0.0001) or 0.0001)

            self.random_state = parameters.get(self.SEED_PARAM, None)

            self.batch_size = parameters.get(self.BATCH_SIZE_PARAM, 'auto')
            self.learning_rate = parameters.get(self.LEARNING_RATE_PARAM, 'constant')
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')
            self.learning_rate_init = float(parameters.get(self.LEARNING_RATE_INIT_PARAM, 0.001))
            self.power_t = float(parameters.get(self.POWER_T_PARAM, 0.5))
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 1))
            self.n_iter_no_change = int(parameters.get(self.N_ITER_NO_CHANGE_PARAM, 10))
            self.momentum = float(parameters.get(self.MOMENTUM_PARAM, 0.9))
            self.nesterovs_momentum = int(parameters.get(self.NESTEROVS_MOMENTUM_PARAM, 1))
            self.early_stopping = int(parameters.get(self.EARLY_STOPPING_PARAM, 0))
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1))
            self.beta_1 = float(parameters.get(self.BETA_1_PARAM, 0.9))
            self.beta_2 = float(parameters.get(self.BETA_2_PARAM, 0.999))
            self.epsilon = float(parameters.get(self.EPSILON_PARAM, 0.00000001))

            vals = [self.alpha, self.max_iter, self.tol]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM, self.TOLERANCE_PARAM]
            for var, att in zip(vals, atts):
                if var < 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>=0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.neural_network import MLPRegressor")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        backup_momentum = float(self.parameters.get(self.MOMENTUM_PARAM, 0.9))
        backup_solver = self.parameters.get(self.SOLVER_PARAM, self.SOLVER_PARAM_ADAM)

        self.shuffle = True if int(self.shuffle) == 1 else False

        self.nesterovs_momentum = True \
            if int(self.nesterovs_momentum) == 1 else False

        self.early_stopping = True if int(self.early_stopping) == 1 else False

        if backup_momentum < 0 or backup_momentum > 1:
            raise ValueError(
                _("Parameter '{}' must be x between 0 and 1 for task {}").format(
                    self.MOMENTUM_PARAM, self.__class__))

        if self.beta_1 < 0 or self.beta_1 >= 1:
            raise ValueError(
                _("Parameter '{}' must be in [0, 1) for task {}").format(
                    self.BETA_1_PARAM, self.__class__))

        if self.beta_2 < 0 or self.beta_2 >= 1:
            raise ValueError(
                _("Parameter '{}' must be in [0, 1) for task {}").format(
                    self.BETA_2_PARAM, self.__class__))

        functions_required = ["""hidden_layer_sizes={hidden_layers}"""
                                  .format(hidden_layers=self.hidden_layers)]

        self.activation = """activation='{activation}'"""\
            .format(activation=self.activation)
        functions_required.append(self.activation)

        self.solver = """solver='{solver}'""".format(solver=self.solver)
        functions_required.append(self.solver)

        self.alpha = """alpha={alpha}""".format(alpha=self.alpha)
        functions_required.append(self.alpha)

        self.max_iter = """max_iter={max_iter}""".format(max_iter=self.max_iter)
        functions_required.append(self.max_iter)

        self.tol = """tol={tol}""".format(tol=self.tol)
        functions_required.append(self.tol)

        if self.random_state is not None:
            self.random_state = """random_state={seed}""".format(seed=self.random_state)
            functions_required.append(self.random_state)

        if type(self.batch_size) == str:
            self.batch_size = """batch_size='{batch_size}'""".format(batch_size=self.batch_size)
        else:
            self.batch_size = """batch_size={batch_size}""".format(
                batch_size=self.batch_size)
        functions_required.append(self.batch_size)

        if self.early_stopping == 1:
            self.validation_fraction = \
                """validation_fraction={validation_fraction}""".format(
                    validation_fraction=self.validation_fraction)
            functions_required.append(self.validation_fraction)

        if backup_solver == 'adam':
            self.beta_1 = """beta_1={beta1}""".format(beta1=self.beta_1)
            functions_required.append(self.beta_1)

            self.beta_2 = """beta_2={beta2}""".format(beta2=self.beta_2)
            functions_required.append(self.beta_2)

            self.epsilon = """epsilon={epsilon}""".format(epsilon=self.epsilon)
            functions_required.append(self.epsilon)

        if backup_solver == 'sgd':
            self.learning_rate = """learning_rate='{learning_rate}'"""\
                .format(learning_rate=self.learning_rate)
            functions_required.append(self.learning_rate)

            self.power_t = """power_t={power_t}"""\
                .format(power_t=self.power_t)
            functions_required.append(self.power_t)

            self.momentum = """momentum={momentum}"""\
                .format(momentum=self.momentum)
            functions_required.append(self.momentum)
            if backup_momentum > 0:
                self.nesterovs_momentum = \
                    """nesterovs_momentum={nesterovs_momentum}""".format(
                        nesterovs_momentum=self.nesterovs_momentum)
            functions_required.append(self.nesterovs_momentum)

        if backup_solver == 'sgd' or backup_solver == 'adam':
            self.learning_rate_init = \
                """learning_rate_init={learning_rate_init}""".format(
                    learning_rate_init=self.learning_rate_init)
            functions_required.append(self.learning_rate_init)

            self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
            functions_required.append(self.shuffle)

            self.n_iter_no_change = """n_iter_no_change={n_iter_no_change}"""\
                .format(n_iter_no_change=self.n_iter_no_change)
            functions_required.append(self.n_iter_no_change)

            self.early_stopping = """early_stopping={early_stopping}"""\
                .format(early_stopping=self.early_stopping)
            functions_required.append(self.early_stopping)

        self.add_functions_required = ', '.join(functions_required)

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = """
                from sklearn.neural_network import MLPRegressor
                from juicer.scikit_learn.util import get_X_train_data, get_label_data
                {output_data} = {input_data}{copy_code}
                X_train = get_X_train_data({input_data}, {columns})
                y = get_label_data({input_data}, {label})
    
                {model} = MLPRegressor({add_functions_required})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(copy_code=copy_code,
                           output_data=self.output,
                           prediction=self.prediction,
                           columns=self.features,
                           model=self.model,
                           input_data=self.input_port,
                           label=self.label,
                           add_functions_required=self.add_functions_required)
            return dedent(code)


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
    CRITERION_PARAM = 'criterion'
    MIN_WEIGHT_FRACTION_LEAF_PARAM = 'min_weight_fraction_leaf'
    MAX_LEAF_NODES_PARAM = 'max_leaf_nodes'
    MIN_IMPURITY_DECREASE_PARAM = 'min_impurity_decrease'
    BOOTSTRAP_PARAM = 'bootstrap'
    OOB_SCORE_PARAM = 'oob_score'
    N_JOBS_PARAM = 'n_jobs'
    RANDOM_STATE_PARAM = 'random_state'
    PREDICTION_PARAM = 'prediction'
    LABEL_PARAM = 'label'
    FEATURES_PARAM = 'features'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.RandomForestRegressor'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.n_estimators = int(parameters.get(self.N_ESTIMATORS_PARAM, 100) or 100)
            self.max_features = parameters.get(self.MAX_FEATURES_PARAM, 'auto') or 'auto'
            self.max_depth = parameters.get(self.MAX_DEPTH_PARAM, None)
            self.min_samples_split = int(parameters.get(self.MIN_SPLIT_PARAM, 2) or 2)
            self.min_samples_leaf = int(parameters.get(self.MIN_LEAF_PARAM, 1) or 1)
            self.criterion = parameters.get(self.CRITERION_PARAM, 'mse') or 'mse'
            self.min_weight_fraction_leaf = float(parameters.get(self.MIN_WEIGHT_FRACTION_LEAF_PARAM, 0) or 0)
            self.max_leaf_nodes = parameters.get(self.MAX_LEAF_NODES_PARAM, None)
            self.min_impurity_decrease = float(parameters.get(self.MIN_IMPURITY_DECREASE_PARAM, 0) or 0)
            self.bootstrap = int(parameters.get(self.BOOTSTRAP_PARAM, 1) or 1)
            self.oob_score = int(parameters.get(self.OOB_SCORE_PARAM, 1) or 1)
            self.n_jobs = int(parameters.get(self.N_JOBS_PARAM, 0) or 0)
            self.random_state = parameters.get(self.RANDOM_STATE_PARAM, None)
            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            vals = [self.n_estimators, self.min_samples_split,
                    self.min_samples_leaf]
            atts = [self.N_ESTIMATORS_PARAM, self.MIN_SPLIT_PARAM,
                    self.MIN_LEAF_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.input_treatment()

            self.transpiler_utils.add_import(
                    "from sklearn.ensemble import RandomForestRegressor")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.n_jobs < -1:
            raise ValueError(
                _("Parameter '{}' must be x >= -1 for task {}").format(
                    self.N_JOBS_PARAM, self.__class__))

        self.n_jobs = 1 if int(self.n_jobs) == 0 else int(self.n_jobs)

        self.bootstrap = True if int(self.bootstrap) == 1 else False

        self.oob_score = True if int(self.oob_score) == 1 else False

        if self.max_depth is not None:
            self.max_depth = int(self.max_depth)
            if self.max_depth <= 0:
                raise ValueError(
                    _("Parameter '{}' must be x>0 or None for task {}").format(
                        self.MAX_DEPTH_PARAM, self.__class__))
        else:
            self.max_depth = None

        if self.max_leaf_nodes is not None and self.max_leaf_nodes != '0':
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            self.max_leaf_nodes = None

        if self.random_state is not None:
            self.random_state = int(self.random_state)
            if self.random_state < 0:
                raise ValueError(
                    _("Parameter '{}' must be x>=0 or None for task {}").format(
                        self.RANDOM_STATE_PARAM, self.__class__))
        else:
            self.random_state = None

        if 0 > self.min_weight_fraction_leaf \
            or self.min_weight_fraction_leaf > 0.5:
            raise ValueError(
                    _("Parameter '{}' must be x >= 0 and x <= 0.5 for "
                      "task {}").format(self.MIN_WEIGHT_FRACTION_LEAF_PARAM,
                                        self.__class__))

        if self.min_impurity_decrease < 0:
            raise ValueError(
                _("Parameter '{}' must be x>=0 or None for task {}").format(
                    self.MIN_IMPURITY_DECREASE_PARAM, self.__class__))

    def generate_code(self):
        if self.has_code:

            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = dedent("""
                {output_data} = {input_data}{copy_code}
                X_train = get_X_train_data({input_data}, {features})
                y = get_label_data({input_data}, {label})
    
                {model} = RandomForestRegressor(n_estimators={n_estimators}, 
                        max_features='{max_features}', 
                        max_depth={max_depth}, 
                        min_samples_split={min_samples_split}, 
                        min_samples_leaf={min_samples_leaf}, 
                        random_state={random_state},
                        n_jobs={n_jobs}, criterion='{criterion}', 
                        min_weight_fraction_leaf={min_weight_fraction_leaf},
                        max_leaf_nodes={max_leaf_nodes}, 
                        min_impurity_decrease={min_impurity_decrease}, 
                        bootstrap={bootstrap},
                        oob_score={oob_score}, warm_start=False)
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """).format(copy_code=copy_code,
                            n_estimators=self.n_estimators,
                            max_features=self.max_features,
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            random_state=self.random_state,
                            model=self.model,
                            n_jobs=self.n_jobs,
                            input_data=self.input_port,
                            output_data=self.output,
                            prediction=self.prediction,
                            criterion=self.criterion,
                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                            max_leaf_nodes=self.max_leaf_nodes,
                            min_impurity_decrease=self.min_impurity_decrease,
                            bootstrap=self.bootstrap,
                            oob_score=self.oob_score,
                            features=self.features,
                            label=self.label)

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
    SEED_PARAM = 'random_state'
    FEATURES_PARAM = 'features'
    LABEL_PARAM = 'label'
    PREDICTION_PARAM = 'prediction'
    POWER_T_PARAM = 'power_t'
    EARLY_STOPPING = 'early_stopping'
    VALIDATION_FRACTION_PARAM = 'validation_fraction'
    LOSS_PARAM = 'loss'
    EPSILON_PARAM = 'epsilon'
    N_ITER_NO_CHANGE_PARAM = 'n_iter_no_change'
    PENALTY_PARAM = 'penalty'
    FIT_INTERCEPT_PARAM = 'fit_intercept'
    ETA0_PARAM = 'eta0'
    AVERAGE_PARAM = 'average'
    LEARNING_RATE_PARAM = 'learning_rate'
    SHUFFLE_PARAM = 'shuffle'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters, named_inputs,
                                     named_outputs)
        self.parameters = parameters
        self.name = 'regression.SGDRegressor'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'train input data', 'input_data_{}'.format(self.order))

        if self.has_code:
            self.add_functions_required = ""
            self.alpha = float(parameters.get(self.ALPHA_PARAM, 0.0001) or 0.0001)
            self.l1_ratio = float(parameters.get(self.ELASTIC_PARAM, 0.15) or 0.15)
            self.max_iter = int(parameters.get(self.MAX_ITER_PARAM, 1000) or 1000)
            self.seed = parameters.get(self.SEED_PARAM, None)

            self.tol = parameters.get(self.TOLERANCE_PARAM, 0.001) or 0.001
            self.tol = abs(float(self.tol))

            self.features = parameters['features']
            self.label = parameters.get(self.LABEL_PARAM, None)
            self.prediction = self.parameters.get(self.PREDICTION_PARAM, 'prediction')

            self.power_t = float(parameters.get(self.POWER_T_PARAM, 0.5))
            self.early_stopping = int(parameters.get(self.EARLY_STOPPING, 0))
            self.validation_fraction = float(parameters.get(self.VALIDATION_FRACTION_PARAM, 0.1))
            self.loss = parameters.get(self.LOSS_PARAM, 'squared_loss')
            self.epsilon = float(parameters.get(self.EPSILON_PARAM, 0.1))
            self.n_iter_no_change = int(parameters.get(self.N_ITER_NO_CHANGE_PARAM, 5))
            self.penalty = parameters.get(self.PENALTY_PARAM, 'l2')
            self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_PARAM, 1))
            self.eta0 = float(parameters.get(self.ETA0_PARAM, 0.01))
            self.average = int(parameters.get(self.AVERAGE_PARAM, 1))
            self.learning_rate = parameters.get(self.LEARNING_RATE_PARAM, 'invscaling')
            self.shuffle = int(parameters.get(self.SHUFFLE_PARAM, 1))

            vals = [self.alpha, self.max_iter,
                    self.n_iter_no_change, self.eta0]
            atts = [self.ALPHA_PARAM, self.MAX_ITER_PARAM,
                    self.N_ITER_NO_CHANGE_PARAM, self.ETA0_PARAM]
            for var, att in zip(vals, atts):
                if var <= 0:
                    raise ValueError(
                            _("Parameter '{}' must be x>0 for task {}").format(
                                    att, self.__class__))

            self.transpiler_utils.add_import(
                    "from sklearn.linear_model import SGDRegressor")
            self.transpiler_utils.add_custom_function(
                    'get_X_train_data', get_X_train_data)
            self.transpiler_utils.add_custom_function(
                    'get_label_data', get_label_data)
            self.input_treatment()

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        self.early_stopping = True if int(self.early_stopping) == 1 else False

        self.shuffle = True if int(self.shuffle) == 1 else False

        if self.seed is not None:
            self.seed = int(self.seed)
            if self.seed < 0:
                raise ValueError(
                    _("Parameter '{}' must be x >= 0 for task {}").format(
                        self.SEED_PARAM, self.__class__))
        else:
            self.seed = None

        if self.l1_ratio < 0 or self.l1_ratio > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.ELASTIC_PARAM, self.__class__))

        if self.validation_fraction < 0 or self.validation_fraction > 1:
            raise ValueError(
                _("Parameter '{}' must be 0 <= x =< 1 for task {}").format(
                    self.VALIDATION_FRACTION_PARAM, self.__class__))

        functions_required = ["""loss='{loss}'""".format(loss=self.loss)]
        self.power_t = """power_t={power_t}""".format(power_t=self.power_t)
        functions_required.append(self.power_t)
        self.early_stopping = """early_stopping={early_stopping}"""\
            .format(early_stopping=self.early_stopping)
        functions_required.append(self.early_stopping)
        self.n_iter_no_change = """n_iter_no_change={n_iter_no_change}"""\
            .format(n_iter_no_change=self.n_iter_no_change)
        functions_required.append(self.n_iter_no_change)
        self.penalty = """penalty='{penalty}'""".format(penalty=self.penalty)
        functions_required.append(self.penalty)
        self.fit_intercept = """fit_intercept={fit_intercept}"""\
            .format(fit_intercept=self.fit_intercept)
        functions_required.append(self.fit_intercept)
        self.average = """average={average}""".format(average=self.average)
        functions_required.append(self.average)
        self.learning_rate = """learning_rate='{learning_rate}'"""\
            .format(learning_rate=self.learning_rate)
        functions_required.append(self.learning_rate)
        self.shuffle = """shuffle={shuffle}""".format(shuffle=self.shuffle)
        functions_required.append(self.shuffle)
        self.alpha = """alpha={alpha}""".format(alpha=self.alpha)
        functions_required.append(self.alpha)
        self.l1_ratio = """l1_ratio={l1_ratio}""".format(l1_ratio=self.l1_ratio)
        functions_required.append(self.l1_ratio)
        self.max_iter = """max_iter={max_iter}""".format(max_iter=self.max_iter)
        functions_required.append(self.max_iter)
        self.seed = """random_state={seed}""".format(seed=self.seed)
        functions_required.append(self.seed)

        if self.loss != 'squared_loss':
            self.epsilon = """epsilon={epsilon}""".format(epsilon=self.epsilon)
            functions_required.append(self.epsilon)
        if self.learning_rate != 'optimal':
            self.eta0 = """eta0={eta0}""".format(eta0=self.eta0)
            functions_required.append(self.eta0)
        if self.early_stopping == 1:
            self.validation_fraction = \
                """validation_fraction={validation_fraction}""".format(
                        validation_fraction=self.validation_fraction)
            functions_required.append(self.validation_fraction)

        self.add_functions_required = ',\n    '.join(functions_required)

    def generate_code(self):
        if self.has_code:

            copy_code = ".copy()" \
                if self.parameters['multiplicity']['train input data'] > 1 else ""

            code = dedent("""
                {output_data} = {input_data}{copy_code}
                X_train = get_X_train_data({input_data}, {columns})
                y = get_label_data({input_data}, {label})
    
                {model} = SGDRegressor({add_functions_required})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """).format(copy_code=copy_code,
                            output_data=self.output,
                            prediction=self.prediction,
                            columns=self.features,
                            model=self.model,
                            input_data=self.input_port,
                            label=self.label,
                            add_functions_required=self.add_functions_required)
            return code


class GeneralizedLinearRegressionOperation(RegressionOperation):

    FIT_INTERCEPT_ATTRIBUTE_PARAM = 'fit_intercept'
    NORMALIZE_ATTRIBUTE_PARAM = 'normalize'
    COPY_X_ATTRIBUTE_PARAM = 'copy_X'
    N_JOBS_ATTRIBUTE_PARAM = 'n_jobs'
    LABEL_ATTRIBUTE_PARAM = 'labels'
    FEATURES_ATTRIBUTE_PARAM = 'features_atr'
    ALIAS_ATTRIBUTE_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        RegressionOperation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.name = 'regression.GeneralizedLinearRegression'
        self.has_code = len(self.named_inputs) == 1 and any(
            [len(self.named_outputs) >= 1, self.contains_results()])
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

        self.model = self.named_outputs.get(
            'model', 'model_{}'.format(self.order))

        self.input_port = self.named_inputs.get(
            'input data', 'input_data_{}'.format(self.order))

        self.fit_intercept = int(parameters.get(self.FIT_INTERCEPT_ATTRIBUTE_PARAM, 1))
        self.normalize = int(parameters.get(self.NORMALIZE_ATTRIBUTE_PARAM, 0))
        self.copy_X = int(parameters.get(self.COPY_X_ATTRIBUTE_PARAM, 1))
        self.n_jobs = int(parameters.get(self.N_JOBS_ATTRIBUTE_PARAM, 0))
        self.features_atr = parameters['features_atr']
        self.label = parameters.get(self.LABEL_ATTRIBUTE_PARAM, None)
        self.alias = self.parameters.get(self.ALIAS_ATTRIBUTE_PARAM, 'prediction')
        if not all([self.LABEL_ATTRIBUTE_PARAM in parameters]):
            msg = _("Parameters '{}' must be informed for task {}")
            raise ValueError(msg.format(
                self.LABEL_ATTRIBUTE_PARAM,
                self.__class__.__name__))

        self.input_treatment()

        self.transpiler_utils.add_import(
                "from sklearn.linear_model import LinearRegression")
        self.transpiler_utils.add_custom_function(
                'get_X_train_data', get_X_train_data)
        self.transpiler_utils.add_custom_function(
                'get_label_data', get_label_data)

    @property
    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=', '):
        return sep.join([self.output, self.model])

    def input_treatment(self):
        if self.n_jobs < -1:
            raise ValueError(
                _("Parameter '{}' must be x>=-1 for task {}").format(
                    self.N_JOBS_ATTRIBUTE_PARAM, self.__class__))

        self.n_jobs = 1 if int(self.n_jobs) == 0 else int(self.n_jobs)

        self.fit_intercept = True if int(self.fit_intercept) == 1 else False

        self.normalize = True if int(self.normalize) == 1 else False

        self.copy_X = True if int(self.copy_X) == 1 else False

    def generate_code(self):
        if self.has_code:
            """Generate code."""
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            code = """
                {output_data} = {input_data}{copy_code}
                X_train = get_X_train_data({input_data}, {columns})
                y = get_label_data({input_data}, {label})
    
                if {fit_intercept}:
                    {model} = LinearRegression(
                        fit_intercept={fit_intercept}, normalize={normalize}, 
                        copy_X={copy_X}, n_jobs={n_jobs})
                else:
                    {model} = LinearRegression(
                        fit_intercept={fit_intercept}, copy_X={copy_X}, 
                        n_jobs={n_jobs})
                {model}.fit(X_train, y)          
                {output_data}['{prediction}'] = {model}.predict(X_train).tolist()
                """.format(copy_code=copy_code,
                           fit_intercept=self.fit_intercept,
                           normalize=self.normalize,
                           copy_X=self.copy_X,
                           n_jobs=self.n_jobs,
                           model=self.model,
                           input_data=self.input_port,
                           label=self.label,
                           output_data=self.output,
                           prediction=self.alias,
                           columns=self.features_atr)

            return dedent(code)
