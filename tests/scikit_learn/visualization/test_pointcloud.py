from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go


import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# pointcloud

import pdb;pdb.set_trace()
    
df = util.iris_polars()

@pytest.fixture
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'pointcloud',
        'display_legend': "LEFT",
        "x": [{
            "binning": "EQUAL_INTERVAL",
            "bins": 30,
            "binSize": 10,
            "emptyBins": "ZEROS",
            "multiplier": 10,
            "decimal_places": 3,
            "prefix": None,
            "suffix": None,
            "label": None,
            "max_displayed": "",
            "group_others": True,
            "sorting": "Y_ASC",
            "attribute": "petalwidth",
            "displayLabel": "TESTE"
        },{
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
            "attribute": "class"
        }],
        "palette": [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#66a61e",
            "#e6ab02",
            "#a6761d",
            "#666666"
        ],
        "y": [{
            "attribute": "petallength",
            "aggregation": "COUNT",
            "compute": None,
            "displayOn": "left",
            "multiplier": 10,
            "decimal_places": 3,
            "prefix": "",
            "suffix": None,
            "label": "TESTE",
            "strokeSize": 0,
            "stroke": None,
            "color": "#da1616",
            "marker": None,
            "custom_color": False,
            "line_color": "#141414",
            "enabled": True
        }],
        "x_axis": {
            "lowerBound": "1",
            "upperBound": "10",
            "logScale": True,
            "display": False,
            "displayLabel": False,
            "label": "TESTE",
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

fig = px.scatter_3d(df_pandas, x=df_pandas['sepallength'], y=df_pandas['sepalwidth'], z=df_pandas['petalwidth'],
              color=df_pandas['class'])

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']
'''

#data tests    
# Teste para verificar o campo 'mode' 
def test_data_mode(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]  
    mode = scatter_data.get('mode')
    assert mode is not None, "Campo 'mode' não encontrado no objeto de dados"
    assert mode == 'markers', "Valor do campo 'mode' incorreto"

# Teste para verificar o campo 'x' 
def test_data_x(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0] 
    x = scatter_data.get('x')
    assert x is not None, "Campo 'x' não encontrado no objeto de dados"

# Teste para verificar o campo 'y' 
def test_data_y(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]  
    y = scatter_data.get('y')
    assert y is not None, "Campo 'y' não encontrado no objeto de dados"
    

# Teste para verificar o campo 'z' 
def test_data_z(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]  
    z = scatter_data.get('z')
    assert z is not None, "Campo 'z' não encontrado no objeto de dados"
   
# Teste para verificar o campo 'type' 
def test_data_type(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]  
    chart_type = scatter_data.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado no objeto de dados"
    assert chart_type == 'scatter3d', "Valor do campo 'type' incorreto"

#test layout

# Teste para verificar o campo 'template' 
def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get('template')
    assert template is not None, "Campo 'template' não encontrado no objeto de layout"

# Teste para verificar o campo 'showlegend' 
def test_layout_showlegend(generated_chart):
    data, layout = generated_chart
    showlegend = layout.get('showlegend')
    assert showlegend is not None, "Campo 'showlegend' não encontrado no objeto de layout"
    assert showlegend == True, "Valor do campo 'showlegend' incorreto"

# Teste para verificar o campo 'title' 
def test_layout_title(generated_chart):
    data, layout = generated_chart
    title = layout.get('title')
    assert title is not None, "Campo 'title' não encontrado no objeto de layout"
    title_text = title.get('text')
    assert title_text is not None, "Campo 'text' dentro de 'title' não encontrado no objeto de layout"
    assert title_text == '', "Valor do campo 'text' dentro de 'title' incorreto"

# Teste para verificar o campo 'scene' no objeto 
def test_layout_scene(generated_chart):
    data, layout = generated_chart
    scene = layout.get('scene')
    assert scene is not None, "Campo 'scene' não encontrado no objeto de layout"
    

# Teste para verificar o campo 'legend' 
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get('legend')
    assert legend is not None, "Campo 'legend' não encontrado no objeto de layout"
    

# Teste para verificar o campo 'margin' 
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get('margin')
    assert margin is not None, "Campo 'margin' não encontrado no objeto de layout"

# Teste para verificar o campo 'xaxis' 
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get('xaxis')
    assert xaxis is not None, "Campo 'xaxis' não encontrado no objeto de layout"


