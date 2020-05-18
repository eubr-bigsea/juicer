from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AddColumnsOperation
import pytest

# Add columns operation
def test_add_columns_success():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['petallength', 'petalwidth', 'class'], 
	slice_size)]
    
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(), 
                          dict([left_df, right_df]))
    assert result['out'].equals(util.iris(size=slice_size))

# def test_add_columns_fail_different_row_number():
#     assert False
# 
# def test_add_columns_fail_missing_parameters():
#     assert False
# 
# def test_add_columns_fail_same_column_names():
#     assert False
# 
# def test_add_columns_success_using_prefix():
#     assert False
# 
# 
