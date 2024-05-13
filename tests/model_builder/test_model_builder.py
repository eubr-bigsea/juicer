import ast
import hashlib
import sys

from io import StringIO
from textwrap import dedent

import pytest

from juicer.meta.meta_minion import MetaMinion
from juicer.meta.operations import (
    LinearRegressionOperation,
    NaiveBayesClassifierOperation,
    KMeansOperation,
    GBTRegressorOperation,
    IsotonicRegressionOperation,
    GeneralizedLinearRegressionOperation,
    DecisionTreeRegressorOperation,
    RandomForestRegressorOperation,
    DecisionTreeClassifierOperation,
    GBTClassifierOperation,
    PerceptronClassifierOperation,
    RandomForestClassifierOperation,
    SVMClassifierOperation,
    LogisticRegressionOperation,
    GaussianMixOperation,
    BisectingKMeansOperation,
    LDAOperation,
    PowerIterationClusteringOperation
)
from juicer.meta.transpiler import ModelBuilderTemplateParams as ModelBuilderParams
from juicer.transpiler import GenerateCodeParams
from juicer.workflow.workflow import Workflow
from tests import compare_ast, format_code_comparison
from tests.scikit_learn import util

from .fixtures import *  # Must be * in order to import fixtures  # noqa: F403


def compute_hyperparameter_count(params):
    result = 1
    for v in params.values():
        if v.get("type") == "list":
            result *= len(v["list"])
        elif v.get("type") == "range":
            result *= v.get("quantity", v.get('size', 1))
    return result


# region Test Sample Operation


@pytest.mark.parametrize("frac", [0.5, 0.1, 0.7, 1.0])
def test_sample_percent_success(builder_params: ModelBuilderParams, frac: float):
    seed = 777
    builder_params.sample.parameters["fraction"] = frac
    builder_params.sample.parameters["seed"] = seed  # Must be set
    builder_params.sample.parameters["type"] = "percent"
    assert (
        f"df = df.sample(withReplacement=False, fraction={frac}, seed={seed})"
        == builder_params.sample.model_builder_code()
    )


@pytest.mark.parametrize("frac", [-200, 24091294, 0, -0.5])
def test_sample_percent_informed_failure(
    builder_params: ModelBuilderParams, frac: float
):
    seed = 777
    builder_params.sample.parameters["fraction"] = frac
    builder_params.sample.parameters["seed"] = seed
    builder_params.sample.parameters["type"] = "percent"
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Parameter 'fraction' must be in range (0, 100)" in str(ve)


@pytest.mark.parametrize("n", [-200, -239523952, 23952935, 0, 5, 100])
def test_sample_head_success(builder_params: ModelBuilderParams, n: int):
    builder_params.sample.parameters["value"] = n
    builder_params.sample.parameters["type"] = "head"
    assert f"df = df.limit({n})" == builder_params.sample.model_builder_code()


@pytest.mark.parametrize("n", [-200, -239523952, 23952935, 0, 5, 100])
def test_sample_fixed_number_success(builder_params: ModelBuilderParams, n: int):
    seed = 123
    builder_params.sample.parameters["value"] = n
    builder_params.sample.parameters["seed"] = seed
    builder_params.sample.parameters["type"] = "value"
    assert (
        f"df = df.sample(False, fraction={n}/df.count(), seed={seed})"
        == builder_params.sample.model_builder_code()
    )


def test_sample_no_type_informed_failure(builder_params: ModelBuilderParams):
    builder_params.sample.parameters["type"] = None
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Parameter 'type' must be informed" in str(ve)


def test_sample_invalid_type_informed_failure(builder_params: ModelBuilderParams):
    builder_params.sample.parameters["type"] = "invalid value"
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Invalid value for parameter 'type'" in str(ve)


def test_no_sample_success(builder_params: ModelBuilderParams):
    seed = 777
    builder_params.sample.parameters["type"] = "percent"
    builder_params.sample.parameters["seed"] = seed
    assert (
        f"df = df.sample(withReplacement=False, fraction=1.0, seed={seed})"
        == builder_params.sample.model_builder_code()
    )


@pytest.mark.parametrize("frac", [0.0, 1.01, -1.0, 10])
def test_sample_invalid_fraction_failure(
    builder_params: ModelBuilderParams, frac: float
):
    builder_params.sample.parameters["type"] = "percent"
    builder_params.sample.parameters["fraction"] = frac
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Parameter 'fraction' must be in range (0, 100)" in str(ve)


# endregion

# region Test Split Operation


def test_split_cross_validation_success(builder_params: ModelBuilderParams):
    # for k in ['strategy', 'seed', 'ratio']:
    #    print(k, builder_params.split.parameters.get(k))

    seed = 12345
    folds = 8
    builder_params.split.strategy = "cross_validation"
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
    ratio = 0.8
    seed = 232
    builder_params.split.strategy = "split"
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


# endregion

# region Evaluator operation


@pytest.mark.parametrize(
    "metric", ["accuracy", "weightedRecall", "weightedPrecision", "f1"]
)
def test_evaluator_multiclass_success(builder_params: ModelBuilderParams, metric: str):
    builder_params.evaluator.task_type = "multiclass-classification"
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


@pytest.mark.parametrize("metric", ["areaUnderROC", "areaUnderPR"])
def test_evaluator_binclass_success(builder_params: ModelBuilderParams, metric: str):
    builder_params.evaluator.task_type = "binary-classification"
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


@pytest.mark.parametrize("metric", ["rmse", "mse", "r2", "mae", "mape"])
def test_evaluator_regression_success(builder_params: ModelBuilderParams, metric: str):
    builder_params.evaluator.task_type = "regression"
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


@pytest.mark.parametrize("metric", ["silhouette"])
def test_evaluator_clustering_success(builder_params: ModelBuilderParams, metric: str):
    builder_params.evaluator.task_type = "clustering"
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
@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_features_supervisioned_with_no_label_failure(
    builder_params: ModelBuilderParams, task_type: str
):
    """Test if type is regression or classification and label informed"""

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = [
        f for f in builder_params.features.features_and_label if f["usage"] != "label"
    ]
    builder_params.features.label = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert "Missing required parameter: label" in str(ve)


@pytest.mark.parametrize("task_type", ["clustering"])
def test_features_unsupervisioned_with_no_label_success(
    builder_params_no_label: ModelBuilderParams, task_type: str
):
    """Test if type is unsupervisioned and label doesn't matter"""

    builder_params_no_label.features.task_type = task_type
    builder_params_no_label.features.process_features_and_label()
    builder_params_no_label.features.model_builder_code()


@pytest.mark.parametrize("task_type", ["regression", "classification", "clustering"])
def test_features_supervisioned_with_no_features_failure(
    builder_params: ModelBuilderParams, task_type: str
):
    """Keep only the label, fail with no feature informed."""

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = [builder_params.features.label]
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert "Missing required parameter: features" in str(ve)


def test_features_invalid_feature_type_failure(builder_params: ModelBuilderParams):
    """Valid feature type: categorical, numerical, textual and vector."""
    builder_params.features.label["feature_type"] = "invalid_type"
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert "Invalid feature type" in str(ve)


def test_features_invalid_feature_usage_failure(builder_params: ModelBuilderParams):
    """Valid feature usage: label, feature, unused (or None)"""
    builder_params.features.label["usage"] = "invalid_usage"
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert "Invalid feature usage" in str(ve)


def test_features_categorical_invalid_transform_failure(
    builder_params: ModelBuilderParams,
):
    """Feature (or label) transform:
        - If type numerical: keep (or None), binarize, quantiles, buckets
        - If type categorical: string_indexer', one_hot_encoder, not_null,
                                hashing
        - If textual: (FIXME: implement)
        - If vector: (FIXME: implement)
    Following tests will test each transform option
    """
    builder_params.features.label["transform"] = "invalid_transform"
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert "Invalid transformation" in str(ve)


def test_label_categorical_success(builder_params: ModelBuilderParams):
    """
    Expect default label setup: remove nulls and use StringIndexer (dummy)
    TODO: Evaluate if SQLTransformer is required,
        because handleInvalid option is equal to 'skip'
    """
    builder_params.features.label["transform"] = "string_indexer"
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
    builder_params: ModelBuilderParams,
):
    """
    TODO: Evaluate if SQLTransformer is required,
        because handleInvalid option is equal to 'skip'
    """
    builder_params.features.label["transform"] = "one_hot_encoder"
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
    builder_params: ModelBuilderParams,
):
    builder_params.features.label["transform"] = "not_null"
    expected_code = dedent("""
        class_na = feature.SQLTransformer(
            statement='SELECT *, INT(ISNULL(class)) AS class_na FROM __THIS__')
        features_stages.append(class_na)
        label = 'class_na'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_label_categorical_handle_null_success(builder_params: ModelBuilderParams):
    builder_params.features.label["missing_data"] = "remove"
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


@pytest.mark.parametrize("constant", ["Iris-virginica", "Iris-setosa"])
def test_label_categorical_handle_null_with_constant_success(
    builder_params: ModelBuilderParams, constant: str
):
    builder_params.features.label["missing_data"] = "constant"
    builder_params.features.label["constant"] = constant
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
    builder_params: ModelBuilderParams,
):
    builder_params.features.label["missing_data"] = "constant"
    builder_params.features.label["constant"] = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert "Missing constant value" in str(ve)


def test_features_numerical_success(builder_params: ModelBuilderParams):
    """
    Change task_type to regression and set sepallength as label
    Expect default label setup: remove nulls?
    """
    builder_params.features.task_type = "regression"
    builder_params.features.features[0]["usage"] = "label"
    builder_params.features.features_and_label.pop()  # remove old label
    builder_params.features.process_features_and_label()

    expected_code = dedent("""
        label = 'sepallength'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_transform_quantis_success(
    builder_params: ModelBuilderParams,
):
    builder_params.features.label["feature_type"] = "numerical"
    builder_params.features.label["transform"] = "quantis"
    builder_params.features.label["quantis"] = 10
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_qtles = feature.QuantileDiscretizer(
            numBuckets=10, inputCol='class',
            outputCol='class_qtles', handleInvalid='skip')
        features_stages.append(class_qtles)
        label = 'class_qtles'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_binarize_success(builder_params: ModelBuilderParams):
    builder_params.features.label["feature_type"] = "numerical"
    builder_params.features.label["transform"] = "binarize"
    builder_params.features.label["threshold"] = 2
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_bin = feature.Binarizer(
            threshold=2, inputCol='class',
            outputCol='class_bin')
        features_stages.append(class_bin)
        label = 'class_bin'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_transform_buckets_success(
    builder_params: ModelBuilderParams,
):
    builder_params.features.label["feature_type"] = "numerical"
    builder_params.features.label["transform"] = "buckets"
    builder_params.features.label["buckets"] = ["0.5", "0.7"]
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_qtles = feature.Bucketizer(
            splits=[-float('inf'), 0.5, 0.7, float('inf')],
            inputCol='class',
            outputCol='class_bkt', handleInvalid='skip')
        features_stages.append(class_qtles)
        label = 'class_bkt'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_transform_invalid_failure(
    builder_params: ModelBuilderParams,
):
    builder_params.features.label["transform"] = "invalid"
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert "Invalid transformation" in str(ve)


@pytest.mark.parametrize(
    "action, algo",
    [
        ("min_max", "MinMaxScaler"),
        ("max_abs", "MaxAbsScaler"),
        ("standard", "StandardScaler"),
    ],
)
def test_features_numerical_scaler_success(
    builder_params_no_label: ModelBuilderParams, action: str, algo: str
):
    # sepallength
    builder_params_no_label.features.features[0]["scaler"] = action
    assert builder_params_no_label.features.features[0]["name"] == "sepallength"

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
    assert {"sepallength_scl", "sepalwidth", "petallength", "petalwidth"} == set(
        builder_params_no_label.features.features_names
    )
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize("constant", [3.14, 1, -123])
def test_features_numerical_constant_if_null_success(
    builder_params_no_label: ModelBuilderParams, constant: float
):
    # sepallength
    builder_params_no_label.features.features[0]["missing_data"] = "constant"

    builder_params_no_label.features.features[0]["constant"] = constant

    expected_code = dedent(f"""
        sepallength_na = feature.SQLTransformer(
            statement=('SELECT *, COALESCE(sepallength, {constant}) '
                       'AS sepallength_na FROM __THIS__'))
        features_stages.append(sepallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert {"sepallength_na", "sepalwidth", "petallength", "petalwidth"} == set(
        builder_params_no_label.features.features_names
    )
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize("action", ["media", "median"])
def test_features_numerical_media_or_median_if_null_success(
    builder_params_no_label: ModelBuilderParams, action: str
):
    # petallength
    builder_params_no_label.features.features[2]["missing_data"] = action

    expected_code = dedent(f"""
        petallength_imp = feature.Imputer(
            strategy='{action}', inputCols=['petallength'],
            outputCols=['petallength_na'])
        features_stages.append(petallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert {"sepallength", "sepalwidth", "petallength_na", "petalwidth"} == set(
        builder_params_no_label.features.features_names
    )
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_median_if_null_multiple_success(
    builder_params_no_label: ModelBuilderParams,
):
    # sepallength and sepalwidth
    builder_params_no_label.features.features[0]["missing_data"] = "median"
    builder_params_no_label.features.features[1]["missing_data"] = "median"

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
    assert {"sepallength_na", "sepalwidth_na", "petallength", "petalwidth"} == set(
        builder_params_no_label.features.features_names
    )

    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_remove_if_null_success(
    builder_params_no_label: ModelBuilderParams,
):
    # sepallength
    builder_params_no_label.features.features[0]["missing_data"] = "remove"

    expected_code = dedent("""
        sepallength_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (sepallength) IS NOT NULL')
        features_stages.append(sepallength_del_nulls)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert {"sepallength", "sepalwidth", "petallength", "petalwidth"} == set(
        builder_params_no_label.features.features_names
    )
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


# TODO Test textual and vector features
def test_textual_tokenize_hash(builder_params: ModelBuilderParams):
    builder_params.features.label["feature_type"] = "textual"
    builder_params.features.label["transform"] = "token_hash"
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_tkn = feature.Tokenizer(
            inputCol='class',
            outputCol='class_tkn')
        features_stages.append(class_tkn)

        class_tkn = feature.HashingTF(
            inputCol='class_tkn',
            outputCol='class_tkn_hash')
        features_stages.append(class_tkn_hash)
        label = 'class_tkn_hash'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_textual_tokenize_stop_hash(builder_params: ModelBuilderParams):
    builder_params.features.label["feature_type"] = "textual"
    builder_params.features.label["transform"] = "token_stop_hash"
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_tkn = feature.StopWordsRemover(
            inputCol='class',
            outputCol='class_stop')
        features_stages.append(class_stop)

        class_tkn = feature.Tokenizer(
            inputCol='class_stop',
            outputCol='class_stop_tkn')
        features_stages.append(class_stop_tkn)

        class_tkn = feature.HashingTF(
            inputCol='class_stop_tkn',
            outputCol='class_stop_tkn_hash')
        features_stages.append(class_stop_tkn_hash)
        label = 'class_stop_tkn_hash'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_textual_count_vectorizer(builder_params: ModelBuilderParams):
    builder_params.features.label["feature_type"] = "textual"
    builder_params.features.label["transform"] = "count_vectorizer"
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_tkn = feature.CountVectorizer(
            inputCol='class',
            outputCol='class_count_vectorizer')
        features_stages.append(class_count_vectorizer)
        label = 'class_count_vectorizer'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_textual_word2vect(builder_params: ModelBuilderParams):
    builder_params.features.label["feature_type"] = "textual"
    builder_params.features.label["transform"] = "word_2_vect"
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement="SELECT * FROM __THIS__ WHERE (class) IS NOT NULL")
        features_stages.append(class_del_nulls)

        class_tkn = feature.Word2Vec(
            inputCol='class',
            outputCol='class_word2vect')
        features_stages.append(class_word2vect)
        label = 'class_word2vect'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


# endregion


# region Estimator Operation
def test_naive_bayes_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "naive bayes"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    nb = NaiveBayesClassifierOperation(params, {}, {})
    assert nb.name == "NaiveBayes"
    assert nb.var == "nb_classifier"

    print(nb.generate_code())
    print(nb.generate_hyperparameters_code())
    print(nb.generate_random_hyperparameters_code())


def test_naive_bayes_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    model_types = ["multinomial", "gaussian"]
    name = "naive bayes"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
        "model_type": {"type": "list", "list": model_types, "enabled": True},
        "smoothing": {
            "type": "range",
            "list": [0.0, 1],
            "enabled": True,
            "quantity": 4,
        },
        "weight_attribute": {
            "type": "list",
            "list": ["species", "class"],
            "enabled": True,
        },
        #'thresholds': {'type': 'list', 'list': ['test'], 'enabled': True}
    }
    nb = NaiveBayesClassifierOperation(params, {}, {})

    expected_code = dedent(f"""
        grid_nb_classifier = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [nb_classifier] }})
            .addGrid(nb_classifier.modelType, {model_types})
            .addGrid(nb_classifier.smoothing, np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(nb_classifier.weightCol, ['species', 'class'])
            .build()
        )""")
    code = nb.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    # Removendo a comparação para parâmetros vazios
    assert nb.get_hyperparameter_count() == 16
    print(nb.generate_random_hyperparameters_code())


'''
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
            .addGrid(nb_classifier.smoothing, np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(nb_classifier.weightCol, ['species', 'class'])
            .build()
        )""")
    code = nb.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    # 4 x 2 x 2 x 1 = 16 parameters
    assert nb.get_hyperparameter_count() == 16
    print(nb.generate_random_hyperparameters_code())
'''


def test_linear_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "linear regression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    lr = LinearRegressionOperation(params, {}, {})
    assert lr.name == "LinearRegression"
    assert lr.var == "linear_reg"

    print(lr.generate_code())
    print(lr.generate_hyperparameters_code())
    print(lr.generate_random_hyperparameters_code())


def test_linear_regression_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364
    name = "linear regression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
        #"aggregation_depth": {"type": "list", "list": [2, 4, 6, 9]},
        "elastic_net": {
            "type": "range",
            "min": 0,
            "max": 1,
            "quantity": 7,
            "distribution": "log_uniform",
        },
        "epsilon": {"type": "list", "list": [2]},
        "tolerance": {
            "type": "range",
            "list": [0.0, 1],
            "enabled": True,
            "quantity": 4,
        },
        # "weight_attribute": {
        #     "type": "list",
        #     "list": ["species", "class"],
        #     "enabled": True,
        # },
        #'thresholds': {'type': 'list', 'list': ['test'], 'enabled': True}
    }
    lr = LinearRegressionOperation(params, {}, {})
    # .addGrid(linear_reg.modelType, {model_types})
    expected_code = dedent("""
        grid_linear_reg = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [linear_reg] })
            #.addGrid(linear_reg.aggregationDepth, [2, 4, 6, 9])
            .addGrid(linear_reg.elasticNetParam,
                np.logspace(np.log10(1e-10), np.log10(1), 7).tolist())
            .addGrid(linear_reg.epsilon, [2.0])
            .addGrid(linear_reg.tol, np.linspace(0, 3, 4, dtype=int).tolist())
            .build()
        )""")

    code = lr.generate_hyperparameters_code()

    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert lr.get_hyperparameter_count() == compute_hyperparameter_count(params)
    print(lr.generate_random_hyperparameters_code())


def test_kmeans_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "linear regression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    km = KMeansOperation(params, {}, {})
    assert km.name == "KMeans"
    assert km.var == "kmeans"

    # print(km.generate_code())
    # print(km.generate_hyperparameters_code())
    # print(km.generate_random_hyperparameters_code())


def test_kmeans_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364
    name = 'k-means clustering'
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
        "aggregation_depth": {"type": "list", "list": [2, 4, 6, 9]},
        "k1": {
            "type": "range",
            "min": 0,
            "max": 1,
            "size": 6,
            "distribution": "log_uniform",
        },
        "number_of_clusters": {"type": "list", "list": [4, 10]},
        "tol": {"type": "list", "list": [2]},
        "type": {"type": "list", "list": ["kmeans", "bisecting"]},
        "init_mode": {"type": "list", "list": ["random", "k-means||"], "enabled": True},
        "max_iterations": {
            "type": "range",
            "list": [0.0, 1],
            "enabled": True,
            "quantity": 4,
        },
        "distance": {"type": "list", "list": ["euclidean"], "enabled": True},
        "seed": {"type": "list", "list": [2]},
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
    print(km.generate_random_hyperparameters_code())

# test of generate code for clustering alg
def test_gaussian_mix_operation_no_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'GaussianMixture'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        }
    }
    gm = GaussianMixOperation(params, {}, {})
    assert gm.name == 'GaussianMixture'
    assert gm.var == 'gaussian_mix'
    '''
    print(gm.generate_code())
    print(gm.generate_hyperparameters_code())
    print(gm.generate_random_hyperparameters_code())
    '''
def test_gaussian_mix_operation_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'GaussianMixture'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'number_of_clusters': {'type': 'list', 'list': [4, 10]},
        'tol': {'type': 'list', 'list': [2]},
        'max_iterations': {'type': 'range', 'list': [0.0, 1], 'enabled': True,
                      'quantity': 4},
        'seed': {'type': 'list', 'list': [2]},
    }
    gm = GaussianMixOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_kmeans = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gaussian_mix] }})
            .addGrid(gaussian_mix.k, [4, 10])
            .addGrid(gaussian_mix.maxIter , np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(gaussian_mix.seed, [2])
            .build()
        )""")
    code = gm.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gm.get_hyperparameter_count() == 6

    print(code)
    print(gm.generate_random_hyperparameters_code())


def test_lda_operation_no_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'LDA'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        }
    }
    lda = LDAOperation(params, {}, {})
    assert lda.name == 'LDA'
    assert lda.var == 'lda'
    '''
    print(gm.generate_code())
    print(gm.generate_hyperparameters_code())
    print(gm.generate_random_hyperparameters_code())
    '''

def test_lda_operation_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'LDA'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'number_of_clusters': {'type': 'list', 'list': [4, 10]},
        'max_iterations': {'type': 'range', 'list': [0.0, 1], 'enabled': True,
                      'quantity': 4},
        'weight_col': 'weight',
        'features': 'features',
        'seed': {'type': 'list', 'list': [2]},
        'checkpoint_interval': 10,
        'optimizer':{'type': 'list', 'list': ['online'], 'enabled': True},
        #'optimizer': {'type': 'boolean', 'enabled': True},
        'learning_offset':{'type': 'float', 'enabled': True},
        'learningDecay':{'type': 'float', 'enabled': True},
        'subsampling_rate': 0.05,
        'optimize_doc_concentration':{'type': 'boolean', 'enabled': True},
        'doc_concentration':{'type': 'float', 'enabled': True},
        'topic_concentration':{'type': 'float', 'enabled': True},
        'topic_distribution_col': 'topicDistribution',
        'keep_last_checkpoint':{'type': 'float', 'enabled': True}

    }
    lda = LDAOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_kmeans = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gaussian_mix] }})
            .addGrid(gaussian_mix.k, [4, 10])
            .addGrid(gaussian_mix.maxIter , np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(gaussian_mix.seed, [2])

            .build()
        )""")
    code = lda.generate_hyperparameters_code()
    print(code)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert lda.get_hyperparameter_count() == 6

    print(code)
    print(lda.generate_random_hyperparameters_code())

def test_pic_operation_no_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'LDA'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        }
    }
    pic = PowerIterationClusteringOperation(params, {}, {})
    assert pic.name == 'PIC'
    assert pic.var == 'pic'
    '''
    print(gm.generate_code())
    print(gm.generate_hyperparameters_code())
    print(gm.generate_random_hyperparameters_code())
    '''

def test_pic_operation_hyperparams_success():
    task_id = '123143-3411-23cf-233'
    operation_id = 2364
    name = 'PIC'
    params = {
        'workflow': {'forms': {}},
        'task': {
            'id': task_id,
            'name': name,
            'operation': {'id': operation_id}
        },
        'number_of_clusters': {'type': 'list', 'list': [4, 10]},
        'init_mode': {'type': 'list', 'list': ['random', 'degree||'],
                       'enabled': True},
        'max_iterations': {'type': 'range', 'list': [0.0, 1], 'enabled': True,
                      'quantity': 4},
        'weight': 'weight',
    }
    pic = PowerIterationClusteringOperation(params, {}, {})
    expected_code = dedent(f"""
        grid_kmeans = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [gaussian_mix] }})
            .addGrid(pic.k, [4, 10])
            .addGrid(pic.initMode, ['random', 'degree||'])
            .addGrid(pic.maxIter , np.linspace(0, 3, 4, dtype=int).tolist())
            .addGrid(pic.weightCol, ['weight'])  # Correção aqui
            .build()
        )""")
    import pdb; pdb.set_trace()
    code = pic.generate_hyperparameters_code()
    print(code)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert pic.get_hyperparameter_count() == 6

    print(pic.generate_random_hyperparameters_code())

# TODO: test all estimators (classifiers, regressors, cluster types)
# endregion


# region regression tests
def test_GBT_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "GBT"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    gbt = GBTRegressorOperation(params, {}, {})
    assert gbt.name == "GBTRegressor"
    assert gbt.var == "gbt_reg"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_gbt_regressor_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "cache_node_ids": {"type": "list", "list": [True, False, True, True]},
        "checkpoint_interval": {
            "type": "list",
            "list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        "feature_subset_strategy": {"type": "list", "list": ["all", "sqrt"]},
        "impurity": {"type": "list", "list": ["variance", "squared", "absolute"]},
        "leaf_col": {"type": "list", "list": ["leaf"]},
        "loss_type": {"type": "list", "list": ["squared"]},
        "max_bins": {"type": "list", "list": [1, 10, 50, 100]},
        "max_depth": {"type": "list", "list": [2, 4, 6, 9]},
        "max_iter": {"type": "list", "list": [1, 10, 50, 100]},
    }

    gbt_reg = GBTRegressorOperation(params, {}, {})
    expected_code = dedent("""
        grid_gbt_reg = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [gbt_reg] })
            .addGrid(gbt_reg.cacheNodeIds, [True, False, True, True])
            .addGrid(gbt_reg.checkpointInterval, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .addGrid(gbt_reg.featureSubsetStrategy, ['all', 'sqrt'])
            .addGrid(gbt_reg.impurity, ['variance', 'squared', 'absolute'])
            .addGrid(gbt_reg.leafCol, ['leaf'])
            .addGrid(gbt_reg.lossType, ['squared'])
            .addGrid(gbt_reg.maxBins, [1, 10, 50, 100])
            .addGrid(gbt_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(gbt_reg.maxIter, [1, 10, 50, 100])
            .build()
        )""")

    code = gbt_reg.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gbt_reg.get_hyperparameter_count() == (
        compute_hyperparameter_count(params))
    print(gbt_reg.generate_random_hyperparameters_code())


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
        'max_depth': {'type': 'list', 'list': [2, 4, 6, 9]},
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
            .addGrid(gbt_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(gbt_reg.maxIter, [1, 10, 50, 100])
            .build()
        )""")

    code = gbt_reg.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gbt_reg.get_hyperparameter_count() == 10
    print(gbt_reg.generate_random_hyperparameters_code())
'''


def test_isotonic_regression_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "isotonic": {"type": "list", "list": [True, False]},
        "weight": {"type": "string", "enabled": True},
    }

    isotonic_reg = IsotonicRegressionOperation(params, {}, {})
    expected_code = dedent("""
        grid_isotonic_reg = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [isotonic_reg] })
            .addGrid(isotonic_reg.isotonic, [True, False])
            .build()
        )""")

    code = isotonic_reg.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert isotonic_reg.get_hyperparameter_count() == (
        compute_hyperparameter_count(params))
    print(isotonic_reg.generate_random_hyperparameters_code())


def test_isotonic_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "IsotonicRegression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    isotonic_reg = IsotonicRegressionOperation(params, {}, {})
    assert isotonic_reg.name == "IsotonicRegression"
    assert isotonic_reg.var == "isotonic_reg"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_generalized_linear_regression_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "family_link": {"type": "list", "list": ["gaussian:identity"]},
        "elastic_net": {
            "type": "range",
            "min": 0,
            "max": 1,
            "size": 8,
            "distribution": "log_uniform",
        },
        "solver": {"type": "list", "list": ["normal", "l-bfgs"], "enabled": True},
    }

    glr = GeneralizedLinearRegressionOperation(params, {}, {})
    expected_code = dedent("""
        grid_gen_linear_regression = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [gen_linear_regression] })
            .addGrid(gen_linear_regression.regParam, np.logspace(np.log10(1e-10), np.log10(1), 8).tolist())
            .addGrid(gen_linear_regression.solver, ['normal', 'l-bfgs'])
            .build()
        )""")

    code = glr.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert glr.get_hyperparameter_count() == compute_hyperparameter_count(params)
    print(glr.get_constrained_params())


def test_generalized_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "GeneralizedLinearRegression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    glr = GeneralizedLinearRegressionOperation(params, {}, {})
    assert glr.name == "GeneralizedLinearRegression"
    assert glr.var == "gen_linear_regression"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_decision_tree_regressor_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "max_bins": {"type": "list", "list": [2, 4, 6, 9], "enabled": True},
        "max_depth": {"type": "list", "list": [2, 4, 6, 9], "enabled": True},
        "min_info_gain": {
            "type": "range",
            "min": 0,
            "max": 1,
            "size": 6,
            "distribution": "uniform",
        },
        "min_instances_per_node": {"type": "list", "list": [2]},
    }

    dt_regressor = DecisionTreeRegressorOperation(params, {}, {})
    expected_code = dedent("""
        grid_dt_reg = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [dt_reg] })
            .addGrid(dt_reg.maxBins, [2, 4, 6, 9])
            .addGrid(dt_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(dt_reg.minInfoGain,
                           np.linspace(0, 1, 6, dtype=int).tolist())
            .addGrid(dt_reg.minInstancesPerNode, [2])
            .build()
        )""")

    code = dt_regressor.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert dt_regressor.get_hyperparameter_count() == (
        compute_hyperparameter_count(params)
    )


def test_decision_tree_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "DecisionTreeRegressor"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    dt_regressor = DecisionTreeRegressorOperation(params, {}, {})
    assert dt_regressor.name == "DecisionTreeRegressor"
    assert dt_regressor.var == "dt_reg"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_random_forest_regressor_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "bootstrap": {"type": "list", "list": [True]},
        "cache_node_ids": {"type": "list", "list": [True]},
        "checkpoint_interval": {"type": "list", "list": [10], "enabled": True},
        "feature_subset_strategy": {
            "type": "list",
            "list": ["auto", "all"],
            "enabled": True,
        },
        "impurity": {"type": "list", "list": ["variance"]},
        "leaf_col": {"type": "list", "list": ["leaf"]},
        "max_bins": {"type": "list", "list": [2, 4, 6, 9], "enabled": True},
        "max_depth": {"type": "list", "list": [2, 4, 6, 9], "enabled": True},
        "max_memory_in_m_b": {"type": "list", "list": [1024], "enabled": True},
        "min_info_gain": {
            "type": "range",
            "min": 0,
            "max": 1,
            "size": 5,
            "distribution": "uniform",
        },
        "min_instances_per_node": {"type": "list", "list": [2], "enabled": True},
        "min_weight_fraction_per_node": {
            "type": "list",
            "list": [0.1],
            "enabled": True,
        },
        "num_trees": {"type": "list", "list": [10, 20, 30], "enabled": True},
        "seed": {"type": "list", "list": [123], "enabled": True},
        "subsampling_rate": {"type": "list", "list": [0.8], "enabled": True},
        "weight_col": {"type": "list", "list": ["weight"]},
    }

    rf_regressor = RandomForestRegressorOperation(params, {}, {})
    expected_code = dedent("""
        grid_rand_forest_reg = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [rand_forest_reg] })
            .addGrid(rand_forest_reg.bootstrap, [True])
            .addGrid(rand_forest_reg.cacheNodeIds, [True])
            .addGrid(rand_forest_reg.checkpointInterval, [10])
            .addGrid(rand_forest_reg.featureSubsetStrategy, ['auto', 'all'])
            .addGrid(rand_forest_reg.impurity, ['variance'])
            .addGrid(rand_forest_reg.leafCol, ['leaf'])
            .addGrid(rand_forest_reg.maxBins, [2, 4, 6, 9])
            .addGrid(rand_forest_reg.maxDepth, [2, 4, 6, 9])
            .addGrid(rand_forest_reg.maxMemoryInMB, [1024])
            .addGrid(rand_forest_reg.minInfoGain, np.linspace(0, 1, 5, dtype=int).tolist())
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

    assert (
        rf_regressor.get_hyperparameter_count() == compute_hyperparameter_count(
            params)
    )  # compute_hyperparameter_count(params)


# endregion

# region classifier tests


def test_decision_tree_classifier_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "cache_node_ids": {"type": "list", "list": [True]},
        "checkpoint_interval": {"type": "list", "list": [1], "enabled": True},
        "impurity": {"type": "list", "list": ["entropy", "gini"], "enabled": True},
        "max_bins": {"type": "list", "list": [2], "enabled": True},
        "max_depth": {"type": "list", "list": [0], "enabled": True},
        "min_info_gain": {"type": "list", "list": [1.0], "enabled": True},
        "min_instances_per_node": {"type": "list", "list": [1], "enabled": True},
        "seed": {"type": "list", "list": [123], "enabled": True},
    }

    dt_classifier = DecisionTreeClassifierOperation(params, {}, {})

    expected_code = dedent("""
        grid_decision_tree = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [decision_tree] })
            .addGrid(decision_tree.cacheNodeIds, [True])
            .addGrid(decision_tree.checkpointInterval, [1])
            .addGrid(decision_tree.impurity, ['entropy', 'gini'])
            .addGrid(decision_tree.maxBins, [2])
            .addGrid(decision_tree.maxDepth, [0])
            .addGrid(decision_tree.minInfoGain, [1.0])
            .addGrid(decision_tree.minInstancesPerNode, [1])
            .addGrid(decision_tree.seed, [123])
            .build()
        )""")

    code = dt_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert dt_classifier.get_hyperparameter_count() == (
        compute_hyperparameter_count(params)
    )


def test_decision_tree_classifier_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "DecisionTreeClassifier"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    dt_classifier = DecisionTreeClassifierOperation(params, {}, {})
    assert dt_classifier.name == "DecisionTreeClassifier"
    assert dt_classifier.var == "decision_tree"


def test_gbt_classifier_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2364

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "cache_node_ids": {"type": "list", "list": [False]},
        "checkpoint_interval": {"type": "list", "list": [1], "enabled": True},
        "loss_type": {"type": "list", "list": ["logistic"]},
        "max_bins": {"type": "list", "list": [32], "enabled": True},
        "max_depth": {"type": "list", "list": [5], "enabled": True},
        "max_iter": {"type": "list", "list": [20], "enabled": True},
        "min_info_gain": {"type": "list", "list": [0.0], "enabled": True},
        "min_instances_per_node": {"type": "list", "list": [1], "enabled": True},
        "seed": {"type": "list", "list": [123], "enabled": True},
        "step_size": {"type": "list", "list": [0.1], "enabled": True},
        "subsampling_rate": {"type": "list", "list": [1.0], "enabled": True},
    }

    gbt_classifier = GBTClassifierOperation(params, {}, {})
    expected_code = dedent(
        """
        grid_gbt_classifier = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [gbt_classifier] })
            .addGrid(gbt_classifier.cacheNodeIds, [False])
            .addGrid(gbt_classifier.checkpointInterval, [1])
            .addGrid(gbt_classifier.lossType,['logistic'])
            .addGrid(gbt_classifier.maxBins, [32])
            .addGrid(gbt_classifier.maxDepth, [5])
            .addGrid(gbt_classifier.maxIter, [20])
            .addGrid(gbt_classifier.minInfoGain, [0.0])
            .addGrid(gbt_classifier.minInstancesPerNode, [1])
            .addGrid(gbt_classifier.seed, [123])
            .addGrid(gbt_classifier.stepSize, [0.1])
            .addGrid(gbt_classifier.subsamplingRate, [1.0])
            .build()
        )""".strip()
    )

    code = gbt_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert gbt_classifier.get_hyperparameter_count() == (
        compute_hyperparameter_count(params)
    )


def test_GBT_classifier_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "GBTClassifier"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    gbt_classifier = GBTClassifierOperation(params, {}, {})
    assert gbt_classifier.name == "GBTClassifier"
    assert gbt_classifier.var == "gbt_classifier"


def test_perceptron_classifier_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2365

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "layers": {"type": "list", "list": [4, 5, 4, 3], "enabled": True},
        "block_size": {"type": "list", "list": [128], "enabled": True},
        "max_iter": {"type": "list", "list": [1], "enabled": True},
        "seed": {"type": "list", "list": [1], "enabled": True},
        "solver": {"type": "list", "list": ["l-bfgs", "gd"], "enabled": True},
    }

    perceptron_classifier = PerceptronClassifierOperation(params, {}, {})
    expected_code = dedent("""
        grid_mlp_classifier = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [mlp_classifier] })
            .addGrid(mlp_classifier.layers, [4, 5, 4, 3])
            .addGrid(mlp_classifier.blockSize, [128])
            .addGrid(mlp_classifier.maxIter, [1])
            .addGrid(mlp_classifier.seed, [1])
            .addGrid(mlp_classifier.solver, ['l-bfgs', 'gd'])
            .build()
        )""")

    code = perceptron_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

    assert perceptron_classifier.get_hyperparameter_count() == 2


def test_perceptron_classifier_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "MultilayerPerceptronClassifier"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    perceptron_classifier = PerceptronClassifierOperation(params, {}, {})
    assert perceptron_classifier.name == "MultilayerPerceptronClassifier"
    assert perceptron_classifier.var == "mlp_classifier"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_random_forest_classifier_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2366

    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "operation": {"id": operation_id}},
        "impurity": {"type": "list", "list": ["entropy", "gini"], "enabled": True},
        "cache_node_ids": {"type": "list", "list": [False], "enabled": True},
        "checkpoint_interval": {"type": "list", "list": [10], "enabled": True},
        "feature_subset_strategy": {
            "type": "list",
            "list": ["auto", "all"],
            "enabled": True,
        },
        "max_bins": {"type": "list", "list": [10, 20, 30], "enabled": True},
        "max_depth": {"type": "list", "list": [5, 10, 15], "enabled": True},
        "min_info_gain": {"type": "list", "list": [0.0], "enabled": True},
        "min_instances_per_node": {"type": "list", "list": [1, 2, 3], "enabled": True},
        "num_trees": {"type": "list", "list": [50, 100, 150], "enabled": True},
        "seed": {"type": "list", "list": [123, 456, 789], "enabled": True},
        "subsampling_rate": {"type": "list", "list": [0.8, 0.9, 1.0], "enabled": True},
    }

    random_forest_classifier = RandomForestClassifierOperation(params, {}, {})
    expected_code = dedent("""
        grid_rand_forest_cls = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [rand_forest_cls] })
            .addGrid(rand_forest_cls.impurity, ['entropy', 'gini'])
            .addGrid(rand_forest_cls.cacheNodeIds, [False])
            .addGrid(rand_forest_cls.checkpointInterval, [10])
            .addGrid(rand_forest_cls.featureSubsetStrategy, ['auto', 'all'])
            .addGrid(rand_forest_cls.maxBins, [10, 20, 30])
            .addGrid(rand_forest_cls.maxDepth, [5, 10, 15])
            .addGrid(rand_forest_cls.minInfoGain, [0.0])
            .addGrid(rand_forest_cls.minInstancesPerNode, [1, 2, 3])
            .addGrid(rand_forest_cls.numTrees, [50, 100, 150])
            .addGrid(rand_forest_cls.seed, [123, 456, 789])
            .addGrid(rand_forest_cls.subsamplingRate, [0.8, 0.9, 1.0])
            .build()
        )""")

    code = random_forest_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)
    assert (
        random_forest_classifier.get_hyperparameter_count()
        == compute_hyperparameter_count(params)
    )


def test_random_forest_classifier_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "RandomForestClassifier"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    random_forest_classifier = RandomForestClassifierOperation(params, {}, {})
    assert random_forest_classifier.name == "RandomForestClassifier"
    assert random_forest_classifier.var == "rand_forest_cls"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_svm_classifier_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2367

    params = {
        "task": {
            "workflow": {"forms": {}},
            "id": task_id,
            "operation": {"id": operation_id},
        },
        "max_iter": {"type": "list", "list": [1, 2, 3], "enabled": True},
        "standardization": {"type": "list", "list": [1, 2, 3], "enabled": True},
        "threshold": {"type": "list", "list": [0.1, 0.2, 0.3], "enabled": True},
        "tol": {"type": "list", "list": [0.01, 0.02, 0.03], "enabled": True},
        "weight_attr": {"type": "list", "list": ["attr1", "attr2"], "enabled": True},
    }

    svm_classifier = SVMClassifierOperation(params, {}, {})
    expected_code = dedent("""
        grid_svm_cls = (tuning.ParamGridBuilder()
            .baseOn({pipeline.stages: common_stages + [svm_cls] })
            .addGrid(svm_cls.maxIter, [1, 2, 3])
            .addGrid(svm_cls.standardization, [1, 2, 3])
            .addGrid(svm_cls.threshold, [0.1, 0.2, 0.3])
            .addGrid(svm_cls.tol, [0.01, 0.02, 0.03])
            .addGrid(svm_cls.weightCol, ['attr1', 'attr2'])
            .build()
        )""")

    code = svm_classifier.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code)

    assert svm_classifier.get_hyperparameter_count() == (
        compute_hyperparameter_count(params)
    )


def test_svm_classifier_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "LinearSVC"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    svm_classifier = SVMClassifierOperation(params, {}, {})
    assert svm_classifier.name == "LinearSVC"
    assert svm_classifier.var == "svm_cls"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


def test_logistic_regression_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2368

    params = {
        "task": {
            "workflow": {"forms": {}},
            "id": task_id,
            "operation": {"id": operation_id},
        },
        "weight_col": {"type": "list", "list": ["attr1", "attr2"], "enabled": True},
        "family": {
            "type": "list",
            "list": ["binomial", "multinomial"],
            "enabled": True,
        },
        "aggregation_depth": {"type": "list", "list": [1, 2, 3], "enabled": True},
        "elastic_net_param": {"type": "list", "list": [0.1, 0.2, 0.3], "enabled": True},
        "fit_intercept": {
            "type": "list",
            "list": [
                True,
                False,
            ],
            "enabled": True,
        },
        "max_iter": {"type": "list", "list": [10, 20, 30], "enabled": True},
        "reg_param": {"type": "list", "list": [0.1, 0.2, 0.3], "enabled": True},
        "tol": {"type": "list", "list": [0.001, 0.002, 0.003], "enabled": True},
        "threshold": {"type": "list", "list": [0.1, 0.2, 0.3], "enabled": True},
        "thresholds": {"type": "list", "list": ["test"], "enabled": True},
    }

    lr = LogisticRegressionOperation(params, {}, {})
    expected_code = dedent(
        f"""
        grid_lr = (tuning.ParamGridBuilder()
            .baseOn({{pipeline.stages: common_stages + [lr] }})
            .addGrid(lr.weightCol, ['attr1', 'attr2'])
            .addGrid(lr.family, ['binomial', 'multinomial'])
            .addGrid(lr.aggregationDepth, [1, 2, 3])
            .addGrid(lr.elasticNetParam, [0.1, 0.2, 0.3])
            .addGrid(lr.fitIntercept, [True, False])
            .addGrid(lr.maxIter, [10, 20, 30])
            .addGrid(lr.regParam, {repr(params['reg_param']['list'])})
            .addGrid(lr.tol, [0.001, 0.002, 0.003])
            .addGrid(lr.threshold, [0.1, 0.2, 0.3])
            .addGrid(lr.thresholds, ['test'])
            .build()
        )""".strip()
    )

    code = lr.generate_hyperparameters_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, msg + format_code_comparison(expected_code, code)

    assert lr.get_hyperparameter_count() == compute_hyperparameter_count(params)


def test_svm_logistic_regression_no_hyperparams_success():
    task_id = "123143-3411-23cf-233"
    operation_id = 2359
    name = "LogisticRegression"
    params = {
        "workflow": {"forms": {}},
        "task": {"id": task_id, "name": name, "operation": {"id": operation_id}},
    }
    lr = LogisticRegressionOperation(params, {}, {})
    assert lr.name == "LogisticRegression"
    assert lr.var == "lr"
    """
    print(gbt.generate_code())
    print(gbt.generate_hyperparameters_code())
    print(gbt.generate_random_hyperparameters_code())
    """


# endregion


# region Temporary tests
def test_grid(builder_params: ModelBuilderParams):
    for k in [
        "strategy",
        "random_grid",
        "seed",
        "max_iterations",
        "max_search_time",
        "parallelism",
    ]:
        print(k, builder_params.grid.parameters.get(k))

    builder_params.grid.has_code = True
    builder_params.grid.strategy = "random"
    builder_params.grid.max_iterations = 30
    # Test strategy parameter:
    # - grid: ok
    # - random: ok

    print(builder_params.grid.model_builder_code())


def test_reduction(builder_params: ModelBuilderParams):
    for k in ["method", "number", "k"]:
        print(k, builder_params.reduction.parameters.get(k))
    print(builder_params.reduction.parameters.keys())

    builder_params.reduction.method = "pca"
    builder_params.reduction.has_code = True
    builder_params.reduction.k = 30
    # Test method parameter:
    # - disable: no reduction
    # - pca: Uses pca

    print(builder_params.reduction.model_builder_code())


def test_estimators(builder_params: ModelBuilderParams):
    print("Estimators", builder_params.estimators)
    for estimator in builder_params.estimators:
        for k in ["method", "number", "k"]:
            print(k, estimator.parameters.get(k))
            # print(estimator.parameters.keys())

        estimator.has_code = True
        # Test method parameter:
        # - disable: no estimator
        # - pca: Uses pca
        print("-" * 20)
        estimator.grid_info.get("value")["strategy"] = "random"
        estimator.grid_info.get("value")["max_iterations"] = 32
        print(estimator.model_builder_code())
        print(estimator.generate_hyperparameters_code())
        # print(estimator.generate_random_hyperparameters_code())


def test_builder_params(sample_workflow: dict, builder_params: ModelBuilderParams):
    loader = Workflow(sample_workflow, config, lang="en")
    instances = loader.workflow["tasks"]

    minion = MetaMinion(
        None,
        config=config,
        workflow_id=sample_workflow["id"],
        app_id=sample_workflow["id"],
    )

    job_id = 1
    opt = GenerateCodeParams(
        loader.graph,
        job_id,
        None,
        {},
        ports={},
        state={},
        task_hash=hashlib.sha1(),
        workflow=loader.workflow,
        tasks_ids=list(loader.graph.nodes.keys()),
    )
    instances, _ = minion.transpiler.get_instances(opt)

    builder_params = minion.transpiler.prepare_model_builder_parameters(
        ops=instances.values()
    )

    print(dir(builder_params))
    print(builder_params.read_data.model_builder_code())
    print(builder_params.sample.model_builder_code())
    print(builder_params.split.model_builder_code())
    print(builder_params.evaluator.model_builder_code())
    print(builder_params.features.model_builder_code())
    print(builder_params.reduction.model_builder_code())
    print(builder_params.grid.model_builder_code())


def xtest_generate_run_code_success(
    sample_workflow: dict, builder_params: ModelBuilderParams
):
    job_id = 1
    estimator = builder_params.estimators[0]
    estimator.has_code = True
    estimator.grid_info.get("value")["strategy"] = "random"
    estimator.grid_info.get("value")["strategy"] = "grid"
    estimator.grid_info.get("value")["max_iterations"] = 32

    # import pdb; pdb.set_trace()
    loader = Workflow(sample_workflow, config, lang="en")

    loader.handle_variables({"job_id": job_id})
    out = StringIO()

    minion = MetaMinion(
        None,
        config=config,
        workflow_id=sample_workflow["id"],
        app_id=sample_workflow["id"],
    )

    minion.transpiler.transpile(
        loader.workflow, loader.graph, config, out, job_id, persist=False
    )
    out.seek(0)
    code = out.read()
    with open("/tmp/juicer_app_1_1_1.py", "w") as f:
        f.write(code)

    print(code, file=sys.stderr)

    result = util.execute(code, {"df": "df"})
    print("Executed")
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()


# endregionn
