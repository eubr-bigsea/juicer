import pytest
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from juicer.scikit_learn.clustering_operation import KMeansClusteringOperation, \
    KMeansModelOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_X_train_data


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

# KMeansClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"random_state": 1}, {"random_state": 1}),

    ({"n_clusters": 10, "random_state": 1}, {"n_clusters": 10, "random_state": 1}),

    ({'init': 'random', 'random_state': 1},
     {'init': 'random', 'random_state': 1}),

    ({'max_iter': 120, 'random_state': 1}, {'max_iter': 120, 'random_state': 1}),

    ({'tolerance': 0.0005, 'random_state': 1}, {'tol': 0.0005, 'random_state': 1}),

    ({'n_init': 12, 'random_state': 1}, {'n_init': 12, 'random_state': 1}),

    ({'n_jobs': -1, 'random_state': 1}, {'n_jobs': -1, 'random_state': 1}),

    ({'algorithm': 'full', 'random_state': 1},
     {'algorithm': 'full', 'random_state': 1}),

], ids=["default_params", "n_clusters_param", "init_param", "max_iter_param",
        "tol_param", "n_init_param", "n_jobs_param", "algorithm_param"])
def test_k_means_clustering_params_success(get_df, get_arguments, get_columns,
                                           operation_par, algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = KMeansModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    x_train = get_X_train_data(test_df, get_columns)

    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=1, n_init=10,
                     n_jobs=None, algorithm='auto')

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    test_df['prediction'] = model_1.fit_predict(x_train)
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_k_means_clustering_prediction_param_success(get_df, get_arguments,
                                                     get_columns):
    arguments = get_arguments
    arguments['parameters'].update({"prediction": "success"})
    arguments = util.add_minimum_ml_args(arguments)
    instance = KMeansModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    assert result['out'].columns[len(get_columns)] == 'success'


@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({"type": "Mini-Batch K-Means", "random_state": 1}, {"random_state": 1}),

    ({"type": "Mini-Batch K-Means", "n_init_mb": 4, "random_state": 1},
     {"n_init": 4, "random_state": 1}),

    ({"type": "Mini-Batch K-Means", "batch_size": 110, "random_state": 1},
     {"batch_size": 110, "random_state": 1}),

    ({"type": "Mini-Batch K-Means", "tol": 0.01, "random_state": 1},
     {"tol": 0.01, "random_state": 1}),

    ({"type": "Mini-Batch K-Means", 'max_no_improvement': 12, "random_state": 1},
     {'max_no_improvement': 12, "random_state": 1})

], ids=["type_param", "n_init_mb_param", "batch_size_param", "tol_param",
        "max_no_imrpovement_param"])
def test_mini_batch_k_means_clustering_params_success(get_arguments,
                                                      get_df,
                                                      get_columns,
                                                      operation_par,
                                                      algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = KMeansModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    x_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.0,
                              random_state=1, n_init=3,
                              max_no_improvement=10,
                              batch_size=100)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    test_df['prediction'] = model_1.fit_predict(x_train)
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),
    ("named_inputs", "train input data")
], ids=["missing_output", "missing_input"])
def test_k_means_clustering_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments

    arguments[selector].pop(drop)

    instance = KMeansModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize("par", ["n_clusters", "max_iter"])
def test_k_means_clustering_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        KMeansModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {KMeansClusteringOperation}" in str(val_err.value)
