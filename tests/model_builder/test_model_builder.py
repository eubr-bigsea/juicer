import ast
import hashlib
import sys
from io import StringIO
from textwrap import dedent

import pytest

from juicer.meta.meta_minion import MetaMinion
from juicer.meta.operations import (
    LinearRegressionOperation, NaiveBayesClassifierOperation, KMeansOperation, GBTRegressorOperation, 
    IsotonicRegressionOperation, GeneralizedLinearRegressionOperation, DecisionTreeRegressorOperation,
    RandomForestRegressorOperation,DecisionTreeClassifierOperation,GBTClassifierOperation, 
    PerceptronClassifierOperation, RandomForestClassifierOperation, SVMClassifierOperation, 
    LogisticRegressionOperation)
from juicer.meta.transpiler import (
    ModelBuilderTemplateParams as ModelBuilderParams)
from juicer.transpiler import GenerateCodeParams
from juicer.workflow.workflow import Workflow
from tests import compare_ast, format_code_comparison
from tests.scikit_learn import util

from .fixtures import *  # Must be * in order to import fixtures

# region Test Sample Operation


def test_sample_percent_success(builder_params: ModelBuilderParams):
    frac = 0.5
    seed = 777
    builder_params.sample.parameters['fraction'] = frac
    builder_params.sample.parameters['seed'] = seed  # Must be set
    builder_params.sample.parameters['type'] = 'percent'
    assert (
        f'df = df.sample(withReplacement=False, fraction={frac}, seed={seed})'
        == builder_params.sample.model_builder_code())


def test_sample_head_success(builder_params: ModelBuilderParams):
    n = 120
    builder_params.sample.parameters['value'] = n
    builder_params.sample.parameters['type'] = 'head'
    assert f'df = df.limit({n})' == builder_params.sample.model_builder_code()


def test_sample_fixed_number_success(builder_params: ModelBuilderParams):
    n = 300
    seed = 123
    builder_params.sample.parameters['value'] = n
    builder_params.sample.parameters['seed'] = seed
    builder_params.sample.parameters['type'] = 'value'
    assert (f'df = df.sample(False, fraction={n}/df.count(), seed={seed})' ==
            builder_params.sample.model_builder_code())


def test_sample_no_type_informed_failure(builder_params: ModelBuilderParams):
    builder_params.sample.parameters['type'] = None
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Parameter 'type' must be informed" in str(ve)


def test_sample_invalid_type_informed_failure(
        builder_params: ModelBuilderParams):
    builder_params.sample.parameters['type'] = 'invalid value'
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Invalid value for parameter 'type'" in str(ve)

@pytest.mark.parametrize('frac', [0.0, 1.01, -1.0, 10])
def test_sample_invalid_fraction_failure(builder_params: ModelBuilderParams,
                                         frac: float):
    builder_params.sample.parameters['type'] = 'percent'
    builder_params.sample.parameters['fraction'] = frac
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert 'Parameter \'fraction\' must be in range (0, 100)' in str(ve)
# endregion

# region Test Split Operation


def test_split_cross_validation_success(builder_params: ModelBuilderParams):
    # for k in ['strategy', 'seed', 'ratio']:
    #    print(k, builder_params.split.parameters.get(k))

    seed = 12345
    folds = 8
    builder_params.split.strategy = 'cross_validation'
    builder_params.split.seed = seed
    builder_params.split.folds = folds
    # Test method parameter:
    # - split:
    # - cross_validation: Not implemented :(
    code = builder_params.split.model_builder_code()
    expected_code = dedent(f"""
    executor = CustomTrainValidationSplit(pipeline, evaluator, grid, 
        seed={seed}, strategy='cross_validation', folds={folds})
    """)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)
    print(builder_params.split.model_builder_code())


def test_split_split_strategy_success(builder_params: ModelBuilderParams):
    ratio = .8
    seed = 232
    builder_params.split.strategy = 'split'
    builder_params.split.seed = seed
    builder_params.split.ratio = ratio

    code = builder_params.split.model_builder_code()
    expected_code = dedent(f"""
    train_ratio = {ratio}
    executor = CustomTrainValidationSplit(pipeline, evaluator, grid, 
        train_ratio, seed={seed}, strategy='split')
    """)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_split_cross_val_strategy_success(builder_params: ModelBuilderParams):
    ratio = .8
    seed = 232
    builder_params.split.strategy = 'cross_validation'
    builder_params.split.seed = seed
    builder_params.split.ratio = ratio

    code = builder_params.split.model_builder_code()
    expected_code = dedent(f"""
    executor = CustomTrainValidationSplit(pipeline, evaluator, grid, 
        seed={seed}, strategy='cross_validation', folds=10)
    """)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

# endregion

# region Evaluator operation

@pytest.mark.parametrize('metric', ['accuracy', 'weightedRecall',
                                    'weightedPrecision', 'f1'])
def test_evaluator_multiclass_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'multiclass-classification'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.MulticlassClassificationEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['areaUnderROC', 'areaUnderPR'])
def test_evaluator_binclass_success(builder_params: ModelBuilderParams,
                                    metric: str):
    builder_params.evaluator.task_type = 'binary-classification'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.BinaryClassificationEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['rmse', 'mse', 'r2', 'mae', 'mape'])
def test_evaluator_regression_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'regression'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.RegressionEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['silhouette'])
def test_evaluator_clustering_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'clustering'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.ClusteringEvaluator(metricName='{metric}')
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

# endregion

# region Features Operation
@pytest.mark.parametrize('task_type', ['regression', 'classification'])
def test_features_supervisioned_with_no_label_failure(
        builder_params: ModelBuilderParams, task_type: str):
    """Test if type is regression or classification and label informed"""

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = [
        f for f in 
            builder_params.features.features_and_label if f['usage'] != 'label']
    builder_params.features.label = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert 'Missing required parameter: label' in str(ve)


@pytest.mark.parametrize('task_type', ['clustering'])
def test_features_unsupervisioned_with_no_label_success(
        builder_params_no_label: ModelBuilderParams, task_type: str):
    """Test if type is unsupervisioned and label doesn't matter"""

    builder_params_no_label.features.task_type = task_type
    builder_params_no_label.features.process_features_and_label()
    builder_params_no_label.features.model_builder_code()

@pytest.mark.parametrize('task_type', ['regression', 'classification',
                                       'clustering'])
def test_features_supervisioned_with_no_features_failure(
        builder_params: ModelBuilderParams, task_type: str):
    """Keep only the label, fail with no feature informed. """

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = [builder_params.features.label]
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert 'Missing required parameter: features' in str(ve)


def test_features_invalid_feature_type_failure(
        builder_params: ModelBuilderParams):
    """ Valid feature type: categorical, numerical, textual and vector. """
    builder_params.features.label['feature_type'] = 'invalid_type'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid feature type' in str(ve)


def test_features_invalid_feature_usage_failure(
        builder_params: ModelBuilderParams):
    """ Valid feature usage: label, feature, unused (or None) """
    builder_params.features.label['usage'] = 'invalid_usage'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid feature usage' in str(ve)


def test_features_categorical_invalid_transform_failure(
        builder_params: ModelBuilderParams):
    """ Feature (or label) transform: 
        - If type numerical: keep (or None), binarize, quantiles, buckets
        - If type categorical: string_indexer', one_hot_encoder, not_null, 
                                hashing
        - If textual: (FIXME: implement)
        - If vector: (FIXME: implement)
    Following tests will test each transform option
    """
    builder_params.features.label['transform'] = 'invalid_transform'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid transformation' in str(ve)


def test_label_categorical_success(builder_params: ModelBuilderParams):
    """ 
    Expect default label setup: remove nulls and use StringIndexer (dummy)
    TODO: Evaluate if SQLTransformer is required, 
        because handleInvalid option is equal to 'skip'
    """
    builder_params.features.label['transform'] = 'string_indexer'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_label_categorical_transform_one_hot_success(
        builder_params: ModelBuilderParams):
    """
    TODO: Evaluate if SQLTransformer is required, 
        because handleInvalid option is equal to 'skip'
    """
    builder_params.features.label['transform'] = 'one_hot_encoder'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        class_ohe = feature.OneHotEncoder(inputCol='class_inx',
            outputCol='class_inx_ohe')
        features_stages.append(class_ohe)
        label = 'class_inx_ohe'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_label_categorical_transform_flag_not_null_success(
        builder_params: ModelBuilderParams):
    builder_params.features.label['transform'] = 'not_null'
    expected_code = dedent("""
        class_na = feature.SQLTransformer(
            statement='SELECT *, INT(ISNULL(class)) AS class_na FROM __THIS__')
        features_stages.append(class_na)
        label = 'class_na'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_label_categorical_handle_null_success(
        builder_params: ModelBuilderParams):
    builder_params.features.label['missing_data'] = 'remove'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('constant', ['Iris-virginica', 'Iris-setosa'])
def test_label_categorical_handle_null_with_constant_success(
        builder_params: ModelBuilderParams, constant: str):
    builder_params.features.label['missing_data'] = 'constant'
    builder_params.features.label['constant'] = constant
    expected_code = dedent(f"""
        class_na = feature.SQLTransformer(
            statement="SELECT *, COALESCE(class, '{constant}') "
                "AS class_na FROM __THIS__")
        features_stages.append(class_na)
        class_inx = feature.StringIndexer(
            inputCol='class_na', outputCol='class_na_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_na_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_categorical_handle_null_with_invalid_constant_failure(
        builder_params: ModelBuilderParams):
    builder_params.features.label['missing_data'] = 'constant'
    builder_params.features.label['constant'] = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Missing constant value' in str(ve)


def test_features_numerical_success(builder_params: ModelBuilderParams):
    """ 
    Change task_type to regression and set sepallength as label
    Expect default label setup: remove nulls? 
    """
    builder_params.features.task_type = 'regression'
    builder_params.features.features[0]['usage'] = 'label'
    builder_params.features.features_and_label.pop()  # remove old label
    builder_params.features.process_features_and_label()

    expected_code = dedent("""
        label = 'sepallength'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


# def test_features_numerical_keep_as_is_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_binarize_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_quantis_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_buckets_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_invalid_failure(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


@pytest.mark.parametrize('action, algo',
                         [('min_max', 'MinMaxScaler'),
                          ('max_abs', 'MaxAbsScaler'),
                          ('standard', 'StandardScaler')])
def test_features_numerical_scaler_success(
        builder_params_no_label: ModelBuilderParams, action: str, algo: str):

    # sepallength
    builder_params_no_label.features.features[0]['scaler'] = action
    assert(
        builder_params_no_label.features.features[0]['name'] == 'sepallength')

    expected_code = dedent(f"""
        sepallength_asm = feature.VectorAssembler(
            handleInvalid='skip',
            inputCols=['sepallength'],
            outputCol='sepallength_asm')
        features_stages.append(sepallength_asm)
        sepallength_scl = feature.{algo}(
            inputCol='sepallength_asm',
            outputCol='sepallength_scl')
        features_stages.append(sepallength_scl)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength_scl', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('constant', [3.14, 1, -123])
def test_features_numerical_constant_if_null_success(
        builder_params_no_label: ModelBuilderParams, constant: float):
    # sepallength
    builder_params_no_label.features.features[0]['missing_data'] = 'constant'

    builder_params_no_label.features.features[0]['constant'] = constant

    expected_code = dedent(f"""
        sepallength_na = feature.SQLTransformer(
            statement=('SELECT *, COALESCE(sepallength, {constant}) '
                       'AS sepallength_na FROM __THIS__'))
        features_stages.append(sepallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength_na', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('action', ['media', 'median'])
def test_features_numerical_media_or_median_if_null_success(
        builder_params_no_label: ModelBuilderParams, action: str):
    # petallength
    builder_params_no_label.features.features[2]['missing_data'] = action

    expected_code = dedent(f"""
        petallength_imp = feature.Imputer(
            strategy='{action}', inputCols=['petallength'], 
            outputCols=['petallength_na'])
        features_stages.append(petallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength', 'sepalwidth', 'petallength_na', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_median_if_null_multiple_success(
        builder_params_no_label: ModelBuilderParams):
    # sepallength and sepalwidth
    builder_params_no_label.features.features[0]['missing_data'] = 'median'
    builder_params_no_label.features.features[1]['missing_data'] = 'median'

    expected_code = dedent("""
        sepallength_imp = feature.Imputer(
            strategy='median', inputCols=['sepallength'], 
            outputCols=['sepallength_na'])
        features_stages.append(sepallength_na)
        sepalwidth_imp = feature.Imputer(
            strategy='median', inputCols=['sepalwidth'], 
            outputCols=['sepalwidth_na'])
        features_stages.append(sepalwidth_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength_na', 'sepalwidth_na', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))

    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_remove_if_null_success(
        builder_params_no_label: ModelBuilderParams):

    # sepallength
    builder_params_no_label.features.features[0]['missing_data'] = 'remove'

    expected_code = dedent("""
        sepallength_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (sepallength) IS NOT NULL')
        features_stages.append(sepallength_del_nulls)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

# TODO Test textual and vector features

# endregion

# region Estimator Operation
def test_naive_bayes_no_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2359
    name = 'naive bayes'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        }
    }
    nb = NaiveBayesClassifierOperation(params, {}, {})
    assert nb.name == 'NaiveBayes'
    assert nb.var == 'nb_classifier'

    print(nb.generate_code())
    print(nb.generate_hyperparameters_code())
    print(nb.generate_random_hyperparameters_code())

def test_naive_bayes_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2359
    model_types = ['multinomial', 'gaussian']
    name = 'naive bayes'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'model_type': {'type': 'list', 'list': model_types, 
                       'enabled': True},
        'smoothing': {'type': 'range', 'list': [0.0, 1], 'enabled': True, 
                      'quantity': 4},
        'weight_attribute': {'type': 'list', 'list': ['species', 'class'], 
                             'enabled': True},
        #'thresholds': {'type': 'list', 'list': ['test'], 'enabled': True}
    }
    nb = NaiveBayesClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_nb_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [nb_classifier] }})
            .addGrid(nb_classifier.modelType, {model_types})
            .addGrid(nb_classifier.smoothing, 
                np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(nb_classifier.weightCol, ['species', 'class'])
            .build()
        )""")
    code = nb.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    # 4 x 2 x 2 x 1 = 16 parameters
    assert nb.get_hyperparameter_count() == 16

    print(code)
    print(nb.generate_random_hyperparameters_code())

def test_linear_regression_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    model_types = ['multinomial', 'gaussian']
    name = 'linear regression'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'aggregation_depth': {'type': 'list', 'list': [2, 4, 6, 9]},
        'elastic_net': {'type': 'range', 'min': 0, 'max': 1, 'size': 6, 
                        "distribution": "log_uniform"},
        'epsilon': {'type': 'list', 'list': [2]},
        'model_type': {'type': 'list', 'list': model_types, 
                       'enabled': True},
        'smoothing': {'type': 'range', 'list': [0.0, 1], 'enabled': True, 
                      'quantity': 4},
        'weight_attribute': {'type': 'list', 'list': ['species', 'class'], 
                             'enabled': True},
        #'thresholds': {'type': 'list', 'list': ['test'], 'enabled': True}
    }
    lr = LinearRegressionOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_linear_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [linear_reg] }})
            #.addGrid(linear_reg.modelType, {model_types})
            .addGrid(linear_reg.aggregationDepth, [2, 4, 6, 9])
            .addGrid(linear_reg.elasticNetParam, 
                np.logspace(np.log10(1e-10), np.log10(1), 3).tolist())
            .addGrid(linear_reg.epsilon, [2.0])
            .build()
        )""")
    code = lr.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert lr.get_hyperparameter_count() == 12

    print(code)
    print(lr.generate_random_hyperparameters_code())

def test_kmeans_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'linear regression'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'aggregation_depth': {'type': 'list', 'list': [2, 4, 6, 9]},
        'k1': {'type': 'range', 'min': 0, 'max': 1, 'size': 6, 
                        "distribution": "log_uniform"},
        'number_of_clusters': {'type': 'list', 'list': [4, 10]},
        'tol': {'type': 'list', 'list': [2]},
        'type': {'type': 'list', 'list': ['kmeans', 'bisecting']},
        'init_mode': {'type': 'list', 'list': ['random', 'k-means||'],
                       'enabled': True},
        'max_iterations': {'type': 'range', 'list': [0.0, 1], 'enabled': True, 
                      'quantity': 4},
        'distance': {'type': 'list', 'list': ['euclidean'], 
                             'enabled': True},
        'seed': {'type': 'list', 'list': [2]},
    }
    km = KMeansOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_kmeans = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [kmeans] }})
            .addGrid(kmeans.k, [4, 10])
            .addGrid(kmeans.initMode, ['random', 'k-means||'])
            .addGrid(kmeans.maxIter , np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(kmeans.distanceMeasure, ['euclidean'])
            .addGrid(kmeans.seed, [2])
            .build()
        )""")
    code = km.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert km.get_hyperparameter_count() == 16

    print(code)
    print(km.generate_random_hyperparameters_code())

# TODO: test all estimators (classifiers, regressors, cluster types)
# endregion

# region regression tests
def test_GBT_regression_no_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2359
    name = 'GBT'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        }
    }
    gbt = GBTRegressorOperation(params, {}, {})
    assert gbt.name == 'GBTRegressor'
    assert gbt.var == 'gbt_reg'
    '''
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    '''

def test_gbt_regressor_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'cache_node_ids': {'type': 'boolean', 'enabled': True},
        'checkpoint_interval': {'type': 'int', 'min': 1, 'max': 10, 'enabled': True},
        'feature_subset_strategy': {'type': 'string', 'enabled': True},
        'impurity': {'type': 'string', 'enabled': True},
        'leaf_col': {'type': 'string', 'enabled': True},
        'loss_type': {'type': 'string', 'enabled': True},
        'max_bins': {'type': 'int', 'min': 1, 'max': 100, 'enabled': True},
        'max_depth': {'type': 'int', 'min': 1, 'max': 10, 'enabled': True},
        'max_iter': {'type': 'int', 'min': 1, 'max': 100, 'enabled': True},
    }

    gbt_reg = GBTRegressorOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_gbt_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gbt_reg] }})
            .addGrid(gbt_reg.cacheNodeIds, [True, False])
            .addGrid(gbt_reg.checkpointInterval, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .addGrid(gbt_reg.featureSubsetStrategy, ['all', 'sqrt'])
            .addGrid(gbt_reg.impurity, ['variance', 'squared', 'absolute'])
            .addGrid(gbt_reg.leafCol, ['leaf'])
            .addGrid(gbt_reg.lossType, ['squared', 'absolute'])
            .addGrid(gbt_reg.maxBins, [1, 10, 50, 100])
            .addGrid(gbt_reg.maxDepth, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .addGrid(gbt_reg.maxIter, [1, 10, 50, 100])
            .build()
        )""")
    
    code = gbt_reg.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gbt_reg.get_hyperparameter_count() == 10

    print(code)
    print(gbt_reg.generate_random_hyperparameters_code())


def test_isotonic_regression_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'isotonic': {'type': 'boolean', 'enabled': True},
        'weight': {'type': 'string', 'enabled': True}
    }

    isotonic_reg = IsotonicRegressionOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_isotonic_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [isotonic_reg] }})
            .addGrid(isotonic_reg.isotonic, [True, False])
            .build()
        )""")
    
    code = isotonic_reg.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert isotonic_reg.get_hyperparameter_count() == 1

    print(code)
    print(isotonic_reg.generate_random_hyperparameters_code())


def test_generalized_linear_regression_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'family_link': ['gaussian:identity'],
        'elastic_net': {'type': 'range', 'min': 0, 'max': 1, 'size': 6, 'distribution': 'log_uniform'},
        'solver': {'type': 'list', 'list': ['normal', 'l-bfgs'], 'enabled': True},
    }

    glr = GeneralizedLinearRegressionOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_gen_linear_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gen_linear_reg] }})
            .addGrid(gen_linear_reg.regParam, 
                np.logspace(np.log10(1e-10), np.log10(1), 3).tolist())
            .addGrid(gen_linear_reg.solver, ['normal', 'l-bfgs'])
            .build()
        )""")

    code = glr.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert glr.get_hyperparameter_count() == 2

    print(code)
    print(glr.get_constrained_params())

def test_decision_tree_regressor_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'max_bins': {'type': 'list', 'list': [2, 4, 6, 9], 'enabled': True},
        'max_depth': {'type': 'list', 'list': [2, 4, 6, 9], 'enabled': True},
        'min_info_gain': {'type': 'range', 'min': 0, 'max': 1, 'size': 6, 'distribution': 'uniform'},
        'min_instances_per_node': 2,  
    }

    dt_regressor = DecisionTreeRegressorOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_dt_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [dt_reg] }})
            .addGrid(dt_reg.maxBins, [2, 4, 6, 9])
            .addGrid(dt_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(dt_reg.minInfoGain, np.linspace(0, 1, 6).tolist())
            .build()
        )""")

    code = dt_regressor.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert dt_regressor.get_hyperparameter_count() == 3  
    '''
    print(code)
    print(dt_regressor.get_constrained_params())
    '''

def test_random_forest_regressor_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'bootstrap': True,
        'cache_node_ids': True,
        'checkpoint_interval': 10,
        'feature_subset_strategy': {'type': 'list', 'list': ['auto', 'all'], 'enabled': True},
        'impurity': 'variance',
        'leaf_col': 'leaf',
        'max_bins': {'type': 'list', 'list': [2, 4, 6, 9], 'enabled': True},
        'max_depth': {'type': 'list', 'list': [2, 4, 6, 9], 'enabled': True},
        'max_memory_in_m_b': 1024,
        'min_info_gain': {'type': 'range', 'min': 0, 'max': 1, 'size': 6, 'distribution': 'uniform'},
        'min_instances_per_node': 2,
        'min_weight_fraction_per_node': 0.1,
        'num_trees': {'type': 'list', 'list': [10, 20, 30], 'enabled': True},
        'seed': 123,
        'subsampling_rate': 0.8,
        'weight_col': 'weight'
    }

    rf_regressor = RandomForestRegressorOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_rand_forest_reg = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [rand_forest_reg] }})
            .addGrid(rand_forest_reg.bootstrap, [True])
            .addGrid(rand_forest_reg.cacheNodeIds, [True])
            .addGrid(rand_forest_reg.checkpointInterval, [10])
            .addGrid(rand_forest_reg.featureSubsetStrategy, ['auto', 'all'])
            .addGrid(rand_forest_reg.impurity, ['variance'])
            .addGrid(rand_forest_reg.leafCol, ['leaf'])
            .addGrid(rand_forest_reg.maxBins, [2, 4, 6, 9])
            .addGrid(rand_forest_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(rand_forest_reg.maxMemoryInMB, [1024])
            .addGrid(rand_forest_reg.minInfoGain, np.linspace(0, 1, 6).tolist())
            .addGrid(rand_forest_reg.minInstancesPerNode, [2])
            .addGrid(rand_forest_reg.minWeightFractionPerNode, [0.1])
            .addGrid(rand_forest_reg.numTrees, [10, 20, 30])
            .addGrid(rand_forest_reg.seed, [123])
            .addGrid(rand_forest_reg.subsamplingRate, [0.8])
            .addGrid(rand_forest_reg.weightCol, ['weight'])
            .build()
        )""")

    code = rf_regressor.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert rf_regressor.get_hyperparameter_count() == 16 
    '''
    print(code)
    print(rf_regressor.get_constrained_params())
    '''

# endregion

# region classifier tests

def test_decision_tree_classifier_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'cache_node_ids': {'type': 'boolean', 'enabled': True},
        'checkpoint_interval': {'type': 'int', 'min': 1, 'enabled': True},
        'impurity': {'type': 'list', 'list': ['entropy', 'gini'], 'enabled': True},
        'max_bins': {'type': 'int', 'min': 2, 'enabled': True},
        'max_depth': {'type': 'int', 'min': 0, 'enabled': True},
        'min_info_gain': {'type': 'float', 'enabled': True},
        'min_instances_per_node': {'type': 'int', 'min': 1, 'enabled': True},
        'seed': {'type': 'int', 'enabled': True},
    }

    dt_classifier = DecisionTreeClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_decision_tree = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [decision_tree] }})
            .addGrid(decision_tree.cacheNodeIds, [True])
            .addGrid(decision_tree.checkpointInterval, [1])
            .addGrid(decision_tree.impurity, ['entropy', 'gini'])
            .addGrid(decision_tree.maxBins, [2])
            .addGrid(decision_tree.maxDepth, [0])
            .addGrid(decision_tree.minInfoGain, [1.0])  
            .addGrid(decision_tree.minInstancesPerNode, [1])
            .addGrid(decision_tree.seed, [None])  
            .build()
        )""")

    code = dt_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert dt_classifier.get_hyperparameter_count() == 8
    '''
    print(code)
    print(dt_classifier.get_constrained_params())
    '''
    
def test_gbt_classifier_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'cache_node_ids': {'type': 'boolean', 'enabled': True},
        'checkpoint_interval': {'type': 'int', 'min': 1, 'enabled': True},
        'loss_type': {'type': 'list', 'list': 'str', 'enabled': True}, 
        'max_bins': {'type': 'int', 'enabled': True},
        'max_depth': {'type': 'int', 'enabled': True},
        'max_iter': {'type': 'int', 'enabled': True},
        'min_info_gain': {'type': 'float', 'enabled': True},
        'min_instances_per_node': {'type': 'int', 'enabled': True},
        'seed': {'type': 'int', 'enabled': True},
        'step_size': {'type': 'float', 'enabled': True},  
        'subsampling_rate': {'type': 'float', 'enabled': True}  
    }

    gbt_classifier = GBTClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_gbt_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gbt_classifier] }})
            .addGrid(gbt_classifier.cacheNodeIds, [True])
            .addGrid(gbt_classifier.checkpointInterval, [1])
            .addGrid(gbt_classifier.lossType) 
            .addGrid(gbt_classifier.maxBins, [1])  
            .addGrid(gbt_classifier.maxDepth, [1])  
            .addGrid(gbt_classifier.maxIter, [1])  
            .addGrid(gbt_classifier.minInfoGain, [1.0])  
            .addGrid(gbt_classifier.minInstancesPerNode, [1])  
            .addGrid(gbt_classifier.seed, [1])  
            .addGrid(gbt_classifier.stepSize, [1.0])  
            .addGrid(gbt_classifier.subsamplingRate, [1.0])  
            .build()
        )""")


    code = gbt_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gbt_classifier.get_hyperparameter_count() == 10  
    '''
    print(code)
    print(gbt_classifier.get_constrained_params())
    '''

def test_perceptron_classifier_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2365

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'layers': {'type': 'string', 'value': '10,5', 'enabled': True},
        'block_size': {'type': 'int', 'enabled': True},
        'max_iter': {'type': 'int', 'enabled': True},
        'seed': {'type': 'int', 'enabled': True},
        'solver': {'type': 'list', 'list': ['l-bfgs', 'gd'], 'enabled': True}
    }

    perceptron_classifier = PerceptronClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_perceptron_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [perceptron_classifier] }})
            .addGrid(perceptron_classifier.layers, [(10, 5)])  
            .addGrid(perceptron_classifier.blockSize, [1])  
            .addGrid(perceptron_classifier.maxIter, [1])  
            .addGrid(perceptron_classifier.seed, [1])  
            .addGrid(perceptron_classifier.solver, ['l-bfgs', 'gd']) 
            .build()
        )""")

    code = perceptron_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert perceptron_classifier.get_hyperparameter_count() == 5
    '''
    print(code)
    print(perceptron_classifier.get_constrained_params())
    '''

def test_random_forest_classifier_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2366

    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'impurity': {'type': 'list', 'list': ['entropy', 'gini'], 'enabled': True},
        'cache_node_ids': {'type': 'boolean', 'enabled': True},
        'checkpoint_interval': {'type': 'int', 'enabled': True},
        'feature_subset_strategy': {'type': 'list', 'list': ['auto', 'all'], 'enabled': True},
        'max_bins': {'type': 'list', 'list': [10, 20, 30], 'enabled': True},
        'max_depth': {'type': 'list', 'list': [5, 10, 15], 'enabled': True},
        'min_info_gain': {'type': 'float', 'enabled': True},
        'min_instances_per_node': {'type': 'list', 'list': [1, 2, 3], 'enabled': True},
        'num_trees': {'type': 'list', 'list': [50, 100, 150], 'enabled': True},
        'seed': {'type': 'list', 'list': [123, 456, 789], 'enabled': True},
        'subsampling_rate': {'type': 'list', 'list': [0.8, 0.9, 1.0], 'enabled': True}
    }

    random_forest_classifier = RandomForestClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_random_forest_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [random_forest_classifier] }})
            .addGrid(random_forest_classifier.impurity, ['entropy', 'gini'])
            .addGrid(random_forest_classifier.cacheNodeIds, [True, False])
            .addGrid(random_forest_classifier.checkpointInterval, [1])
            .addGrid(random_forest_classifier.featureSubsetStrategy, ['auto', 'all'])
            .addGrid(random_forest_classifier.maxBins, [10, 20, 30])
            .addGrid(random_forest_classifier.maxDepth, [5, 10, 15])
            .addGrid(random_forest_classifier.minInfoGain, [0.0])
            .addGrid(random_forest_classifier.minInstancesPerNode, [1, 2, 3])
            .addGrid(random_forest_classifier.numTrees, [50, 100, 150])
            .addGrid(random_forest_classifier.seed, [123, 456, 789])
            .addGrid(random_forest_classifier.subsamplingRate, [0.8, 0.9, 1.0])
            .build()
        )""")

    code = random_forest_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert random_forest_classifier.get_hyperparameter_count() == 11
    '''
    print(code)
    print(random_forest_classifier.get_constrained_params())
    '''

def test_svm_classifier_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2367

    params = {
        'task': {
            'workflow': {'forms': {}},
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'max_iter': {'type': 'int', 'enabled': True},
        'standardization': {'type': 'int', 'enabled': True},
        'threshold': {'type': 'float', 'enabled': True},
        'tol': {'type': 'float', 'enabled': True},
        'weight_attr': {'type': 'list', 'list': ['attr1', 'attr2'], 'enabled': True}
    }

    svm_classifier = SVMClassifierOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_svm_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [svm_classifier] }})
            .addGrid(svm_classifier.maxIter, [1, 2, 3])  
            .addGrid(svm_classifier.standardization, [1, 2, 3])  
            .addGrid(svm_classifier.threshold, [0.1, 0.2, 0.3])  
            .addGrid(svm_classifier.tol, [0.01, 0.02, 0.03])  
            .addGrid(svm_classifier.weightCol, ['attr1', 'attr2'])  
            .build()
        )""")

    code = svm_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code)

    assert svm_classifier.get_hyperparameter_count() == 5
    '''
    print(code)
    print(svm_classifier.get_constrained_params())
    '''

def test_logistic_regression_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2368

    params = {
        'task': {
            'workflow': {'forms': {}},
            'id': task_id,
            'operation': {'id': operation_id}
        },
        'weight_col': {'type': 'list', 'list': ['attr1', 'attr2'], 'enabled': True},
        'family': {'type': 'list', 'list': ['binomial', 'multinomial'], 'enabled': True},
        'aggregation_depth': {'type': 'int', 'enabled': True},
        'elastic_net_param': {'type': 'float', 'enabled': True},
        'fit_intercept': {'type': 'boolean', 'enabled': True},
        'max_iter': {'type': 'int', 'enabled': True},
        'reg_param': {'type': 'float', 'enabled': True},
        'tol': {'type': 'float', 'enabled': True},
        'threshold': {'type': 'float', 'enabled': True},
        'thresholds': {'type': 'list', 'list': ['test'], 'enabled': True}
    }

    lr = LogisticRegressionOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_lr = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [lr] }})
            .addGrid(lr.weightCol, ['attr1', 'attr2'])  
            .addGrid(lr.family, ['binomial', 'multinomial'])  
            .addGrid(lr.aggregationDepth, [1, 2, 3])  
            .addGrid(lr.elasticNetParam, [0.1, 0.2, 0.3])  
            .addGrid(lr.fitIntercept, [True, False])  
            .addGrid(lr.maxIter, [10, 20, 30]) 
            .addGrid(lr.regParam, [0.01, 0.02, 0.03])  
            .addGrid(lr.tol, [0.001, 0.002, 0.003])  
            .addGrid(lr.threshold, [0.1, 0.2, 0.3])  
            .addGrid(lr.thresholds, ['test'])  
            .build()
        )""")

    code = lr.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code)

    assert lr.get_hyperparameter_count() == 10
    '''
    print(code)
    print(lr.get_constrained_params())
    '''


# endregion

# region Temporary tests
def test_grid(builder_params: ModelBuilderParams):
    for k in ['strategy', 'random_grid', 'seed', 'max_iterations',
              'max_search_time', 'parallelism']:
        print(k, builder_params.grid.parameters.get(k))

    builder_params.grid.has_code = True
    builder_params.grid.strategy = 'random'
    builder_params.grid.max_iterations = 30
    # Test strategy parameter:
    # - grid: ok
    # - random: ok

    print(builder_params.grid.model_builder_code())


def test_reduction(builder_params: ModelBuilderParams):
    for k in ['method', 'number', 'k']:
        print(k, builder_params.reduction.parameters.get(k))
    print(builder_params.reduction.parameters.keys())

    builder_params.reduction.method = 'pca'
    builder_params.reduction.has_code = True
    builder_params.reduction.k = 30
    # Test method parameter:
    # - disable: no reduction
    # - pca: Uses pca

    print(builder_params.reduction.model_builder_code())


def test_estimators(builder_params: ModelBuilderParams):

    print('Estimators',  builder_params.estimators)
    for estimator in builder_params.estimators:
        for k in ['method', 'number', 'k']:
            print(k, estimator.parameters.get(k))
            # print(estimator.parameters.keys())

        estimator.has_code = True
        # Test method parameter:
        # - disable: no estimator
        # - pca: Uses pca
        print('-'*20)
        estimator.grid_info.get('value')['strategy'] = 'random'
        estimator.grid_info.get('value')['max_iterations'] = 32
        print(estimator.model_builder_code())
        print(estimator.generate_hyperparameters_code())
        # print(estimator.generate_random_hyperparameters_code())


def test_builder_params(sample_workflow: dict, builder_params: ModelBuilderParams):

    loader = Workflow(sample_workflow, config, lang='en')
    instances = loader.workflow['tasks']

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    job_id = 1
    opt = GenerateCodeParams(loader.graph, job_id, None, {},
                             ports={}, state={}, task_hash=hashlib.sha1(),
                             workflow=loader.workflow,
                             tasks_ids=list(loader.graph.nodes.keys()))
    instances, _ = minion.transpiler.get_instances(opt)

    builder_params = minion.transpiler.prepare_model_builder_parameters(
        ops=instances.values())

    print(dir(builder_params))
    print(builder_params.read_data.model_builder_code())
    print(builder_params.sample.model_builder_code())
    print(builder_params.split.model_builder_code())
    print(builder_params.evaluator.model_builder_code())
    print(builder_params.features.model_builder_code())
    print(builder_params.reduction.model_builder_code())
    print(builder_params.grid.model_builder_code())


def xtest_generate_run_code_success(sample_workflow: dict, builder_params: ModelBuilderParams):

    job_id = 1
    estimator = builder_params.estimators[0]
    estimator.has_code = True
    estimator.grid_info.get('value')['strategy'] = 'random'
    estimator.grid_info.get('value')['strategy'] = 'grid'
    estimator.grid_info.get('value')['max_iterations'] = 32

    # import pdb; pdb.set_trace()
    loader = Workflow(sample_workflow, config, lang='en')

    loader.handle_variables({'job_id': job_id})
    out = StringIO()

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    minion.transpiler.transpile(loader.workflow, loader.graph,
                                config, out, job_id,
                                persist=False)
    out.seek(0)
    code = out.read()
    with open('/tmp/juicer_app_1_1_1.py', 'w') as f:
        f.write(code)

    print(code, file=sys.stderr)

    result = util.execute(code, {'df': 'df'})
    print('Executed')
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()

# endregion