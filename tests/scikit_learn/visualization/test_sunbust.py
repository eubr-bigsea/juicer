from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# sunburst

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

@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=emit_event)
    code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
    result = util.execute(code, vis_globals)
    generated_chart = result.get('d')
    data = generated_chart['data']
    layout = generated_chart['layout']
    print(data)
    return data,layout


'''
df_pol = df.collect()
df_pandas = df_pol.to_pandas()
valores = len(df_pandas['sex'])

fig = px.sunburst(df_pandas, path=df_pandas['pclass'], values=valores, color=df_pandas['sex'])

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
'''

#test data
# Teste para verificar o campo 'branchvalues' 
def test_data_branchvalues(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    branchvalues = sunburst_data.get('branchvalues')
    assert branchvalues is not None, "Campo 'branchvalues' não encontrado no objeto de dados"
    assert branchvalues == 'total', "Valor do campo 'branchvalues' incorreto"

# Teste para verificar o campo 'customdata' 
def test_data_customdata(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    customdata = sunburst_data.get('customdata')
    assert customdata is not None, "Campo 'customdata' não encontrado no objeto de dados"
    expected_customdata = [[323.0], [277.0], [709.0]]
    assert customdata == expected_customdata, "Valor do campo 'customdata' incorreto"
'''
# Teste para verificar o campo 'customdata' está incorreto
def test_data_customdata02(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    customdata = sunburst_data.get('customdata')
    assert customdata is not None, "Campo 'customdata' não encontrado no objeto de dados"
    expected_customdata = [[32.0], [27.0], [79.0]]
    assert customdata == expected_customdata, "Valor do campo 'customdata' incorreto"'''

# Teste para verificar o campo 'domain' 
def test_data_domain(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    domain = sunburst_data.get('domain')
    assert domain is not None, "Campo 'domain' não encontrado no objeto de dados"
   

# Teste para verificar o campo 'hovertemplate'
def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0] 
    hovertemplate = sunburst_data.get('hovertemplate')
    assert hovertemplate is not None, "Campo 'hovertemplate' não encontrado no objeto de dados"
    

# Teste para verificar o campo 'ids' 
def test_data_ids(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    ids = sunburst_data.get('ids')
    assert ids is not None, "Campo 'ids' não encontrado no objeto de dados"
    expected_ids = ['1st', '2nd', '3rd']
    assert ids == expected_ids, "Valores do campo 'ids' incorretos"
'''
# Teste para verificar o campo 'ids' está incorreto
def test_data_ids02(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    ids = sunburst_data.get('ids')
    assert ids is not None, "Campo 'ids' não encontrado no objeto de dados"
    expected_ids = ['4st', '5nd', '6rd']
    assert ids == expected_ids, "Valores do campo 'ids' incorretos"'''

# Teste para verificar o campo 'labels' 
def test_data_labels(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    labels = sunburst_data.get('labels')
    assert labels is not None, "Campo 'labels' não encontrado no objeto de dados"
    expected_labels = ['1st', '2nd', '3rd']
    assert labels == expected_labels, "Valores do campo 'labels' incorretos"

# Teste para verificar o campo 'marker' 
def test_data_marker(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0] 
    marker = sunburst_data.get('marker')
    assert marker is not None, "Campo 'marker' não encontrado no objeto de dados"
    

# Teste para verificar o campo 'name' 
def test_data_name(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    name = sunburst_data.get('name')
    assert name is not None, "Campo 'name' não encontrado no objeto de dados"
    assert name == '', "Valor do campo 'name' incorreto"

# Teste para verificar o campo 'parents' 
def test_data_parents(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0] 
    parents = sunburst_data.get('parents')
    assert parents is not None, "Campo 'parents' não encontrado no objeto de dados"
    expected_parents = ['', '', '']
    assert parents == expected_parents, "Valores do campo 'parents' incorretos"
'''
# Teste para verificar o campo 'parents' está incorreto
def test_data_parents02(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0] 
    parents = sunburst_data.get('parents')
    assert parents is not None, "Campo 'parents' não encontrado no objeto de dados"
    expected_parents = ['1', '2', '3']
    assert parents == expected_parents, "Valores do campo 'parents' incorretos"'''

# Teste para verificar o campo 'values' 
def test_data_values(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    values = sunburst_data.get('values')
    assert values is not None, "Campo 'values' não encontrado no objeto de dados"
    expected_values = [323.0, 277.0, 709.0]
    assert values == expected_values, "Valores do campo 'values' incorretos"

# Teste para verificar o campo 'type' 
def test_data_type(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    chart_type = sunburst_data.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado no objeto de dados"
    assert chart_type == 'sunburst', "Valor do campo 'type' incorreto"
'''
# Teste para verificar o campo 'type' está incorreto
def test_data_type02(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    chart_type = sunburst_data.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado no objeto de dados"
    assert chart_type == 'heapmap', "Valor do campo 'type' incorreto"'''


#layout tests

# Teste para verificar o campo 'template' 
def test_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    assert layout_template is not None, "Campo 'template' não encontrado no objeto de layout"
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}
    assert layout_template == expected_template, "Valores do campo 'template' incorretos"
'''
# Teste para verificar o campo 'template' esta incorreto 
def test_layout_template02(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    assert layout_template is not None, "Campo 'template' não encontrado no objeto de layout"
    expected_template = {'data': {'box': [{'type': 'box'}]}}
    assert layout_template == expected_template, "Valores do campo 'template' incorretos"'''

# Teste para verificar o campo 'coloraxis' 
def test_layout_coloraxis(generated_chart):
    data, layout = generated_chart
    layout_coloraxis = layout.get('coloraxis')
    assert layout_coloraxis is not None, "Campo 'coloraxis' não encontrado no objeto de layout"
    expected_coloraxis = {
        'colorbar': {'title': {'text': 'count(*)'}},
        'colorscale': [
            [0.0, '#000000'],
            [0.25, '#e60000'],
            [0.5, '#e6d200'],
            [0.75, '#ffffff'],
            [1.0, '#a0c8ff']
        ]
    }
    assert layout_coloraxis == expected_coloraxis, "Valores do campo 'coloraxis' incorretos"

# Teste para verificar o campo 'legend' 
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get('legend')
    assert layout_legend is not None, "Campo 'legend' não encontrado no objeto de layout"
    expected_legend = {'tracegroupgap': 0}
    assert layout_legend == expected_legend, "Valores do campo 'legend' incorretos"
'''
# Teste para verificar o campo 'legend' esta incorreto
def test_layout_legend02(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get('legend')
    assert layout_legend is not None, "Campo 'legend' não encontrado no objeto de layout"
    expected_legend = {'tracegroupgap': 1}
    assert layout_legend == expected_legend, "Valores do campo 'legend' incorretos"'''

# Teste para verificar o campo 'margin' 
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    layout_margin = layout.get('margin')
    assert layout_margin is not None, "Campo 'margin' não encontrado no objeto de layout"
    expected_margin = {'t': 30, 'l': 30, 'r': 30, 'b': 30}
    assert layout_margin == expected_margin, "Valores do campo 'margin' incorretos"

# Teste para verificar o campo 'showlegend' 
def test_layout_showlegend(generated_chart):
    data, layout = generated_chart
    layout_showlegend = layout.get('showlegend')
    assert layout_showlegend is not None, "Campo 'showlegend' não encontrado no objeto de layout"
    expected_layout_showlegend = False
    assert layout_showlegend == expected_layout_showlegend, "Valor do campo 'showlegend' no layout incorreto"
'''
# Teste para verificar o campo 'showlegend' esta incorreto
def test_layout_showlegend02(generated_chart):
    data, layout = generated_chart
    layout_showlegend = layout.get('showlegend')
    assert layout_showlegend is not None, "Campo 'showlegend' não encontrado no objeto de layout"
    expected_layout_showlegend = True
    assert layout_showlegend == expected_layout_showlegend, "Valor do campo 'showlegend' no layout incorreto"'''

# Teste para verificar o campo 'xaxis' 
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    assert layout_xaxis is not None, "Campo 'xaxis' não encontrado no objeto de layout"
    expected_xaxis = {'categoryorder': 'trace'}
    assert layout_xaxis == expected_xaxis, "Valores do campo 'xaxis' incorretos"



def generate_chart_with_args(arguments, dataframe):
    try:
        instance = VisualizationOperation(**arguments)
        vis_globals = dict(iris=dataframe) 
        generated_chart = instance.generate_chart(visual_globals=vis_globals)
        return generated_chart
    except Exception as e:
        # Captura uma exceção 
        raise ValueError(f"Erro ao gerar o gráfico: {str(e)}")

def test_missing_chart_type(get_arguments, get_df):
    # Remova o tipo de gráfico dos argumentos
    arguments = get_arguments
    del arguments['parameters']['type']

    # Tenta gerar o gráfico com parâmetros ausentes
    try:
        result = generate_chart_with_args(arguments, get_df)
    except ValueError as e:
        # Verifica se a exceção foi gerada
        assert "Erro ao gerar o gráfico" in str(e)
        assert "Tipo de gráfico não especificado" in str(e)
    else:
        pytest.fail("Esperava-se que uma exceção fosse gerada")

def test_missing_input_data(get_arguments):
    arguments = get_arguments
    # Defina a base de dados como None para testar o caso onde não é fornecida
    arguments['named_inputs']['input data'] = None

    try:
        result = generate_chart_with_args(arguments, None)  # Passamos None como a base de dados
    except ValueError as e:
        assert "Erro ao gerar o gráfico" in str(e)
        assert "Base de dados não fornecida" in str(e)
    else:
        pytest.fail("Esperava-se que uma exceção fosse gerada quando a base de dados não é fornecida")

def test_missing_color_scale(get_arguments):
    arguments = get_arguments
    # Remoção a escala de cores dos argumentos
    del arguments['parameters']['color_scale']

    try:
        result = generate_chart_with_args(arguments, get_df)  
    except ValueError as e:
        assert "Erro ao gerar o gráfico" in str(e)
        assert "Escala de cores não fornecida" in str(e)
    else:
        pytest.fail("Esperava-se que uma exceção fosse gerada quando a escala de cores não é fornecida")
