from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SelectOperation
import pytest

# Select
# 
def test_select_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth', 
        'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    result = util.execute(instance.generate_code(), 
                          dict([df]))
    assert result['out'].equals(util.iris(size=slice_size))

def test_select_fail_parameters_not_needed():
    #como nao exige parametros deve falhar se fornecer parametro?

def test_select_success_with_2_valid_named_inputs():
    #em vez do input data ser so df[0] seria df[0] e df[1] por exemplo, devendo selecionar as duas colunas

def test_select_fail_invalid_and_valid_named_inputs():
    #deve dar erro se for inserido um named input valido e um invalido ou vice e versa certo?

def test_select_fail_invalid_named_inputs():
    #deve dar erro ao ser inserido um named input invalido

def test_select_fail_2_invalid_named_inputs():
    #deve dar erro ao serem inseridos 2 named input invalidos

def test_select_fail_2_equal_named_inputs():
    #deve dar erro ao inserir 2 named_inputs iguais ou selecionar a mesma coluna 2 vezes?

def test_select_success_missing_input_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo

def test_select_success_missing_output_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo