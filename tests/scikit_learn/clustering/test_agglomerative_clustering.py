import pytest
from sklearn.cluster import AgglomerativeClustering as AGC

from juicer.scikit_learn.clustering_operation import AgglomerativeModelOperation, \
    AgglomerativeClusteringOperation
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
        'parameters': {'attributes': get_columns,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# AgglomerativeClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),

    ({'number_of_clusters': 5}, {"n_clusters": 5}),

    ({'linkage': 'single', 'affinity': 'l1'},
     {"linkage": 'single', "affinity": 'l1'}),

], ids=['default_params', 'number_of_clusters_param',
        'affinity_and_linkage_params'])
def test_agglomerative_clustering_params_success(get_columns, get_arguments,
                                                 get_df, operation_par,
                                                 algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = AgglomerativeModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    x_train = get_X_train_data(test_df, get_columns)
    agg = AGC(n_clusters=2, linkage='ward', affinity='euclidean')

    for key, value in algorithm_par.items():
        setattr(agg, key, value)

    test_df['prediction'] = agg.fit_predict(x_train)
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(agg)


def test_agglomerative_clustering_alias_param_success(get_df, get_arguments,
                                                      get_columns):
    arguments = get_arguments
    arguments['parameters'].update({'alias': 'success'})
    arguments = util.add_minimum_ml_args(arguments)
    instance = AgglomerativeModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': get_df})
    assert result['out'].columns[len(get_columns)] == 'success'


@pytest.mark.parametrize(('selector', 'drop'), [
    ("named_outputs", "output data"),

    ("named_inputs", "input data")

], ids=["missing_output", "missing_input"])
def test_agglomerative_clustering_no_code_success(get_arguments, selector, drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = AgglomerativeModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_agglomerative_clustering_missing_attributes_param_fail(get_arguments):
    arguments = get_arguments
    arguments['parameters'].pop("attributes")
    with pytest.raises(ValueError) as val_err:
        AgglomerativeModelOperation(**arguments)
    assert f"Parameter 'attributes' must be informed for task" \
           f" {AgglomerativeClusteringOperation}" in str(val_err)


def test_agglomerative_clustering_invalid_number_of_clusters_param_fail(
        get_arguments):
    arguments = get_arguments
    arguments['parameters'].update({'number_of_clusters': -1})
    with pytest.raises(ValueError) as val_err:
        AgglomerativeModelOperation(**arguments)
    assert f"Parameter 'number_of_clusters' must be x>0 for task" \
           f" {AgglomerativeClusteringOperation}" in str(val_err)
