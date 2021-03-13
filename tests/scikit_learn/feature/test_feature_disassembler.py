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
    test_df = df.copy()
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureDisassemblerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

# # # # # # # # # # Fail # # # # # # # # # #
