from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import LdaClusteringOperation, \
    LdaClusteringModelOperation, ClusteringModelOperation
from juicer.scikit_learn.util import get_X_train_data
from sklearn.decomposition import LatentDirichletAllocation
import pytest
import pandas as pd


@pytest.fixture
def get_columns():
    return ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']


@pytest.fixture
def get_df(get_columns):
    return util.iris(get_columns)


@pytest.fixture
def get_arguments(get_columns):
    return {
        'parameters': {'features': get_columns,
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# LdaClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"seed": 1}, {"random_state": 1}),

    ({"number_of_topics": 11, "seed": 1},
     {"n_components": 11, "random_state": 1}),

    ({'doc_topic_pior': 0.2, "seed": 1},
     {'doc_topic_prior': 0.2, "random_state": 1}),

    ({'topic_word_prior': 0.2, "seed": 1},
     {'topic_word_prior': 0.2, "random_state": 1}),

    ({'learning_method': 'batch', "seed": 1},
     {'learning_method': 'batch', "random_state": 1}),

    ({'max_iter': 12, "seed": 1}, {'max_iter': 12, "random_state": 1})

], ids=["seed_param", "number_of_topics_param", "doc_topic_pior_param",
        "topic_word_prior_param", "learning_method_param", "max_iter_param"])
def test_lda_clustering_params_success(get_columns, get_df, get_arguments,
                                       operation_par, algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = LdaClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=10,
                                        random_state=1)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    x_train = get_X_train_data(test_df, get_columns)
    model_1.fit(x_train)
    test_df['prediction'] = model_1.transform(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_lda_clustering_prediction_param_success(get_columns, get_df,
                                                 get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"prediction": "success"})

    arguments = util.add_minimum_ml_args(arguments)
    instance = LdaClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df})
    assert result['out'].columns[len(get_columns)] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_lda_clustering_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = LdaClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_lda_clustering_invalid_learning_method_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({"learning_method": "invalid"})
    with pytest.raises(ValueError) as val_err:
        LdaClusteringModelOperation(**arguments)
    assert f"Invalid optimizer value 'invalid' for class" \
           f" {LdaClusteringOperation}" in str(val_err.value)


@pytest.mark.parametrize("par", ["number_of_topics", "max_iter"])
def test_lda_clustering_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        LdaClusteringModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {LdaClusteringOperation}" in str(val_err.value)


def test_lda_clustering_missing_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop("features")
    with pytest.raises(ValueError) as val_err:
        LdaClusteringModelOperation(**arguments)
    assert f"Parameters 'features' must be informed for task" \
           f" LdaClusteringOperation" in str(val_err.value)
