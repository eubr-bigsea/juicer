from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go


import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# Funnel

import pdb;pdb.set_trace()

df = util.funel_polars()

@pytest.fixture
def get_df():
    return util.funel_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'funnel',
            'display_legend': "AUTO",
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
                "attribute": "Stage"
            }],
            "palette": [
                "#1F77B4",
                "#FF7F0E",
                "#2CA02C",
                "#D62728",
                "#9467BD",
                "#8C564B",
                "#E377C2",
                "#7F7F7F",
                "#BCBD22",
                "#17BECF"
            ],
            "y": [{
                "attribute": "Count",
                "aggregation": "MIN",
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

fig = go.Figure(go.Funnel(
    y = df_pandas['Stage'],
    x = df_pandas['Count'],
    textposition = "inside",
    textinfo = "value+percent initial",
    opacity = 0.65, marker = {"color": ["deepskyblue", "lightsalmon", "tan", "teal", "silver"],
    "line": {"width": [4, 2, 2, 3, 1, 1], "color": ["wheat", "wheat", "blue", "wheat", "wheat"]}},
    connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}})
    )

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']
'''


#data tests    
# Teste para verificar se o tipo do gráfico é 'funnel'
def test_funnel_type(generated_chart):
    data, layout = generated_chart
    assert all(item.get('type') == 'funnel' for item in data), "Tipo de gráfico incorreto"

# Teste para verificar se o tipo do gráfico está incorreto
def test_funnel_type2(generated_chart):
    data, layout = generated_chart
    assert all(item.get('type') == 'pie' for item in data), "Tipo de gráfico incorreto"

# Teste para verificar se os valores do eixo x estão corretos
def test_funnel_x_values(generated_chart):
    data, layout = generated_chart
    x_values = data[0].get('x')
    expected_x_values = [13873.0, 10533.0, 5443.0, 2703.0, 908.0]  
    assert x_values == expected_x_values, "Valores do eixo x incorretos"

# Teste para verificar se os valores do eixo x estão incorretos
def test_funnel_x_values2(generated_chart):
    data, layout = generated_chart
    x_values = data[0].get('x')
    expected_x_values = [873.0, 533.0, 43.0, 2703.0, 908.0]  
    assert x_values == expected_x_values, "Valores do eixo x incorretos"

# Teste para verificar se os valores do eixo y estão corretos
def test_funnel_y_values(generated_chart):
    data, layout = generated_chart
    y_values = data[0].get('y')
    expected_y_values = ['Website visit', 'Downloads', 'Potential customers', 'Invoice sent', 'Closed deals']  
    assert y_values == expected_y_values, "Valores do eixo y incorretos"

# Teste para verificar se os valores do eixo y estão incorretos
def test_funnel_y_values2(generated_chart):
    data, layout = generated_chart
    y_values = data[0].get('y')
    expected_y_values = ['Website rate', 'Upload', 'customers', 'Invoice sent', 'Closed deals']  
    assert y_values == expected_y_values, "Valores do eixo y incorretos"

# Teste para verificar se o campo 'textinfo' está definido corretamente no primeiro item de dados
def test_funnel_textinfo(generated_chart):
    data, layout = generated_chart
    textinfo = data[0].get('textinfo')
    expected_textinfo = 'value+percent initial'  
    assert textinfo == expected_textinfo, "Valor do campo 'textinfo' incorreto"


#testlayout

# Teste para verificar se o template do layout está definido corretamente
def test_funnel_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}  
    assert layout_template == expected_template, "Template do layout incorreto"

# Teste para verificar as margens (margin) do layout 
def test_funnel_layout_margin(generated_chart):
    data, layout = generated_chart
    layout_margin = layout.get('margin')
    expected_margin = {'l': 30, 'r': 30, 't': 30, 'b': 30}  
    assert layout_margin == expected_margin, "Margens (margin) do layout incorretas"

# Teste para verificar as configurações do eixo x (xaxis) do layout
def test_funnel_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    expected_xaxis = {'categoryorder': 'trace'}  
    assert layout_xaxis == expected_xaxis, "Configurações do eixo x (xaxis) do layout incorretas"
