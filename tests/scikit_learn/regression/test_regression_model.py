from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import RegressionModelOperation
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# RegressionModel
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_regression_model_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {'multiplicity': {'train input data': 1},
                       'features': [['petalwidth', 'sepalwidth']],
                       'label': [['petalwidth']]},
        'named_inputs': {
            'algorithm': 'LinearRegression()',
            'train input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionModelOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    print(result['out'])
