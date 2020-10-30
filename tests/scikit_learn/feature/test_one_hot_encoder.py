from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import OneHotEncoderOperation
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def hotencoder(df, attr, alias='onehotenc_1'):
    """
    Creates the cipher to encode checking each value on each column/row
    Uses sets so values don't repeat, sort to alphabetic/numeric order
    And finally encode using cipher.
    """
    df_oper = df[attr]
    cipher = {col: sorted({df_oper.loc[idx, col] for idx in
                           df_oper.index}) for col in df_oper.columns}
    result = [
        [1.0 if val == df_oper.loc[idx, col] else 0.0 for col in
         df_oper.columns for val in cipher[col]] for idx in df_oper.index
    ]

    return pd.DataFrame({alias: result})


# OneHotEncoder
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_one_hot_encoder_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['onehotenc_1']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth']))
