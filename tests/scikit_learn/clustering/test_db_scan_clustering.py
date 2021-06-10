import pytest
from sklearn.cluster import DBSCAN

from juicer.scikit_learn.clustering_operation import \
    DBSCANClusteringModelOperation, DBSCANClusteringOperation, \
    ClusteringModelOperation
from tests.scikit_learn import util
from tests.scikit_learn.util import get_X_train_data


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
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


# DBSCANClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),

    ({'eps': 0.2}, {'eps': 0.2}),

    ({'min_samples': 10}, {"min_samples": 10}),

    ({'metric': 'manhattan'}, {"metric": 'manhattan'})

], ids=["default_params", "eps_param", "min_samples_param", "metric_param"])
def test_db_scan_clustering_params_success(get_df,
                                           get_arguments,
                                           get_columns,
                                           operation_par,
                                           algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = DBSCANClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    x_train = get_X_train_data(test_df,
                               get_columns)
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')

    for key, value in algorithm_par.items():
        setattr(dbscan, key, value)

    test_df['prediction'] = dbscan.fit_predict(x_train)
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(dbscan)


def test_db_scan_clustering_prediction_param_success(get_df, get_arguments,
                                                     get_columns):
    arguments = get_arguments
    arguments['parameters'].update({'prediction': 'success'})

    arguments = util.add_minimum_ml_args(arguments)
    instance = DBSCANClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    assert result['out'].columns[len(get_columns)] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_db_scan_clustering_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = DBSCANClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_db_scan_clustering_missing_features_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop("features")
    with pytest.raises(ValueError) as val_err:
        DBSCANClusteringModelOperation(**arguments)
    assert f"Parameter 'features' must be informed for task" \
           f" {ClusteringModelOperation}" in str(val_err.value)


@pytest.mark.parametrize('par', ['min_samples', 'eps'])
def test_db_scan_clustering_invalid_params_fail(get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        DBSCANClusteringModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {DBSCANClusteringOperation}" in str(val_err.value)
