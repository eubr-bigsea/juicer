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
        'parameters': {'sepallength'},
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

def test_select_fail_missing_parameters():
    #deve falhar ao nao receber parametros, não gerar códigou ou gerar uma saída vazia?

def test_select_success_valid_parameters():
    # deve selecionar uma coluna da tabela

def test_select_success_with_2_valid_parameters():
    #deve selecionar duas colunas da tabela

def test_select_fail_invalid_and_valid_parameters():
    #deve dar erro se for inserido um parametro valido e um invalido ou vice e versa certo?

def test_select_fail_invalid_parameter():
    #deve dar erro ao ser inserido um parametro invalido

def test_select_fail_2_invalid_parameters():
    #deve dar erro ao serem inseridos 2 parametros invalidos

def test_select_success_2_equal_parameters():
    #se forem fornecidos 2 parametros iguais deve selecionar a coluna e não tem problema

def test_select_fail_invalid_named_inputs():
    # deve dar erro ao ser inserido um named input invalido

def test_select_fail_2_named_inputs():
    # deve dar erro pois select só aceita um named input

def test_select_success_missing_input_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo, has_code() deve retornar False

def test_select_success_missing_output_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo