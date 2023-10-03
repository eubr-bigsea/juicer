from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors


# vamos usar o sunburst para fazer os testes de erro

import pdb;pdb.set_trace()
    
df = util.titanic_polars()

@pytest.fixture
def get_df():
    return util.titanic_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'sunburst',
        'display_legend': "HIDE",
        "x": [{
            "binning": None,
            "bins": 20,
            "binSize": 10,
            "emptyBins": "ZEROS",
            "multiplier": None,
            "decimal_places": 2,
            "prefix": None,
            "suffix": None,
            "label": None,
            "max_displayed": None,
            "group_others": True,
            "sorting": "NATURAL",
            "attribute": "pclass"
        }],
        "color_scale": [
            "#000000",
            "#e60000",
            "#e6d200",
            "#ffffff",
            "#a0c8ff"
        ],
        "y": [{
            "attribute": "*",
            "aggregation": "COUNT",
            "compute": None,
            "displayOn": "left",
            "multiplier": None,
            "decimal_places": 2,
            "prefix": None,
            "suffix": None,
            "label": None,
            "strokeSize": 0,
            "stroke": None,
            "color": None,
            "marker": None,
            "enabled": True
        }],
        "x_axis": {
           "lowerBound": None,
            "upperBound": None,
            "logScale": False,
            "display": True,
            "displayLabel": True,
            "label": None,
            "multiplier": None,
            "decimal_places": 2,
            "prefix": None,
            "suffix": None
        },
        "y_axis": {
            "lowerBound": None,
            "upperBound": None,
            "logScale": False,
            "display": True,
            "displayLabel": True,
            "label": None,
            "multiplier": None,
            "decimal_places": 2,
            "prefix": None,
            "suffix": None
        },
        "task_id": "0"
        },
        'named_inputs': {
        'input data': "iris",
        },
        'named_outputs': {
        'output data': 'out'
        }
    }
    
def emit_event(*args, **kwargs):
    print(args, kwargs)

def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=emit_event)
    code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
    result = util.execute(code, vis_globals)
    generated_chart_code = result.get('d')
    data = generated_chart_code['data']
    layout = generated_chart_code['layout']
    print(data)
    return data,layout




    
def test_missing_chart_type(get_arguments, get_df):
    # Remove o tipo de gráfico dos argumentos
    arguments = get_arguments
    del arguments['parameters']['type']

    # Tenta gerar o gráfico com parâmetros ausentes
    with pytest.raises(ValueError) as ex:
        result = generated_chart(arguments, get_df)
    assert "Missing required parameter: type" in str(ex.value)
    
def test_missing_input_data(get_arguments, get_df):
    # Simula a ausência de dados definindo a variável 'get_df' como None
    get_df = None

    # Define a base de dados como None para testar o caso onde a base não é fornecida
    arguments = get_arguments
    arguments['named_inputs']['input data'] = get_df

    # Tenta gerar o gráfico com a base de dados ausente
    with pytest.raises(Exception) as ex:
        result = generated_chart(arguments, get_df)

    # Verifica se a mensagem de erro esperada foi lançada, erro gerado pela falta do df
    assert str(ex.value) == "'NoneType' object has no attribute 'clone'"

'''
def test_missing_color_scale(get_arguments, get_df):
    # Remove a escala de cores dos argumentos
    arguments = get_arguments
    #arguments['parameters']['color_scale'] = None 
    del arguments['parameters']['color_scale']

    # Tenta gerar o gráfico com a escala de cores ausente
    with pytest.raises(ValueError) as ex:
        result = generated_chart(arguments, get_df)
    assert "Missing required parameter: color_scale" in str(ex.value)
'''
def test_missing_x(get_arguments, get_df):
    # Remove o parâmetro y dos argumentos
    arguments = get_arguments.copy()
    del arguments['parameters']['x']

    # Tenta gerar o gráfico com parâmetros ausentes, os valores x da base
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: x" in str(ex.value)

def test_missing_y(get_arguments, get_df):
    # Remove o parâmetro y dos argumentos
    arguments = get_arguments.copy()
    del arguments['parameters']['y']

    # Tenta gerar o gráfico com parâmetros ausentes, os valores y da base
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: y" in str(ex.value)


def test_missing_x_axis(get_arguments, get_df):
    # Remove o parâmetro dos valores tradados do eixo x 
    arguments = get_arguments.copy()
    del arguments['parameters']['x_axis']

    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: x_axis" in str(ex.value)

def test_missing_y_axis(get_arguments, get_df):
    # Remove o parâmetro dos valores tradados do eixo y 
    arguments = get_arguments.copy()
    del arguments['parameters']['y_axis']

    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: y_axis" in str(ex.value)


def test_missing_display_legend(get_arguments, get_df):
    # Remove o parâmetro display_legend dos argumentos
    arguments = get_arguments.copy()
    del arguments['parameters']['display_legend']

    # Tenta gerar o gráfico com parâmetros ausentes
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: display_legend" in str(ex.value)
