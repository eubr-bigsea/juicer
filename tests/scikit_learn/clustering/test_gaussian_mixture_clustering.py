import pytest
from sklearn.mixture import GaussianMixture

from juicer.scikit_learn.clustering_operation import \
    GaussianMixtureClusteringOperation, GaussianMixtureClusteringModelOperation
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


# GaussianMixtureClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
@pytest.mark.parametrize(("operation_par", "algorithm_par"), [
    ({}, {}),

    ({'max_iter': 50}, {"max_iter": 50}),

    ({'tol': 0.005}, {"tol": 0.005}),

    ({'n_components': 2, 'random_state': 1},
     {'n_components': 2, 'random_state': 1}),

    ({'covariance_type': 'tied'}, {'covariance_type': 'tied'}),

    ({'reg_covar': 1e-05}, {'reg_covar': 1e-05}),

    ({'n_init': 2}, {'n_init': 2}),

    ({'random_state': 2002}, {'random_state': 2002})

], ids=["default_params", "max_iter_param", "tol_param", "n_components_param",
        "covariance_type_param", "reg_covar_param", "n_init_param",
        "random_state_param"])
def test_gaussian_mixture_clustering_params_success(get_df, get_arguments,
                                                    get_columns, operation_par,
                                                    algorithm_par):
    test_df = get_df
    arguments = get_arguments

    arguments['parameters'].update(operation_par)

    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df})
    x_train = get_X_train_data(test_df, get_columns)
    model_1 = GaussianMixture(n_components=1, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=None)

    for key, value in algorithm_par.items():
        setattr(model_1, key, value)

    test_df['prediction'] = model_1.fit_predict(x_train)
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gaussian_mixture_clustering_prediction_param_success(get_arguments,
                                                              get_df,
                                                              get_columns):
    arguments = get_arguments

    arguments["parameters"].update({"prediction": "success"})

    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': get_df})
    assert result['out'].columns[len(get_columns)] == 'success'


@pytest.mark.parametrize(("selector", "drop"), [
    ("named_outputs", "output data"),

    ("named_inputs", "train input data")

], ids=["missing_output", "missing_input"])
def test_gaussian_mixture_clustering_no_code_success(get_arguments, selector,
                                                     drop):
    arguments = get_arguments
    arguments[selector].pop(drop)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
@pytest.mark.parametrize("par", ["max_iter", "n_components"])
def test_gaussian_mixture_clustering_invalid_n_comps_and_max_iter_params_fail(
        get_arguments, par):
    arguments = get_arguments
    arguments['parameters'].update({par: -1})
    with pytest.raises(ValueError) as val_err:
        GaussianMixtureClusteringModelOperation(**arguments)
    assert f"Parameter '{par}' must be x>0 for task" \
           f" {GaussianMixtureClusteringOperation}" in str(val_err.value)
