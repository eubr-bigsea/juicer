from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import FeatureDisassemblerOperation
from textwrap import dedent
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# FeatureDisassembler
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_feature_disassembler_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    df['vector'] = df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    test_df = df[['vector', 'sepallength', 'sepalwidth', ]].copy()
    test_df.columns = ['vector', 'vector_1', 'vector_2']
    df = df[['vector']]
    arguments = {
        'parameters': {
            'feature': ['vector'],
            'top_n': 2,
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    instance = FeatureDisassemblerOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df': df})
    assert result['out'].equals(test_df)

# # # # # # # # # # Fail # # # # # # # # # #
