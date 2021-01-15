from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation \
    import QuantileDiscretizerOperation as QDO
from sklearn.preprocessing import KBinsDiscretizer
import pytest

# QuantileDiscretizer
#


def test_quantile_discretizer_success():
    slice_size = 10
    columns = ['sepallength']
    df = util.iris(columns, slice_size)
    test_df = df.copy()

    arguments = {
        'parameters': {QDO.ATTRIBUTE_PARAM: columns,
                       QDO.N_QUANTILES_PARAM: 2,
                       'multiplicity': {'input data': 0},
                       QDO.ALIAS_PARAM: "out"
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = QDO(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    model_1 = KBinsDiscretizer(n_bins=2,
                               encode='ordinal',
                               strategy=QDO.DISTRIBUITION_PARAM_QUANTIS)
    X_train = util.get_X_train_data(test_df, columns)

    test_out["out"] = model_1.fit_transform(X_train).flatten().tolist()

    for idx in result['out'].index:
        assert result['out'].loc[idx, "out"] == pytest.approx(
                test_out.loc[idx, "out"], 0.1)
