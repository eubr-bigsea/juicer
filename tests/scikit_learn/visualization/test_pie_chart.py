from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# Grafico de pizza

import pdb;pdb.set_trace()
    
df = util.iris_polars()

@pytest.fixture
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'pie',
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
            "attribute": "class"
        }],
        "color_scale": [
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
            "marker": None
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
# mudança do tipo
df_pol = df.collect()
df_pandas = df_pol.to_pandas()

# Definir uma escala de cores personalizada
custom_colors = plotly.colors.sequential.Viridis
# Gerar o gráfico com a escala de cores personalizada

contagem_valores = len(df_pandas['sepalwidth'])

fig = px.pie(df_pandas, values=df_pandas['sepalwidth'], names=df_pandas['class'])

#fig = px.density_heatmap(df, x=df_select_result, y=df_select_result1, marginal_x="box", marginal_y="violin")

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']
'''

#data tests    
# Teste para verificar o campo 'domain' 
def test_data_domain(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    domain = data_entry.get('domain')
    assert domain is not None, "Campo 'domain' não encontrado em data"
    assert domain == {'x': [0.0, 1.0], 'y': [0.0, 1.0]}, "Valor do campo 'domain' incorreto"

# Teste para verificar o campo 'hovertemplate' 
def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    hovertemplate = data_entry.get('hovertemplate')
    assert hovertemplate is not None, "Campo 'hovertemplate' não encontrado em data"
    assert hovertemplate == 'class=%{label}<br>count(*)=%{value}<extra></extra>', "Valor do campo 'hovertemplate' incorreto"

# Teste para verificar o campo 'labels'
def test_data_labels(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    labels = data_entry.get('labels')
    assert labels is not None, "Campo 'labels' não encontrado em data"
    assert labels == ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], "Valor do campo 'labels' incorreto"

# Teste para verificar o campo 'labels' está incorreto
def test_data_labels02(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    labels = data_entry.get('labels')
    assert labels is not None, "Campo 'labels' não encontrado em data"
    assert labels == ['Iris-setosacm', 'Iris-versicolorcm', 'Iris-virginicacm'], "Valor do campo 'labels' incorreto"

# Teste para verificar o campo 'legendgroup' 
def test_data_legendgroup(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    legendgroup = data_entry.get('legendgroup')
    assert legendgroup == '', "Valor do campo 'legendgroup' incorreto"

# Teste para verificar o campo 'legendgroup' incorreto
def test_data_legendgroup02(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    legendgroup = data_entry.get('legendgroup')
    assert legendgroup == 'qualquer legenda', "Valor do campo 'legendgroup' incorreto"

# Teste para verificar o campo 'showlegend' 
def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    showlegend = data_entry.get('showlegend')
    assert showlegend is not None, "Campo 'showlegend' não encontrado em data"
    assert showlegend is True, "Valor do campo 'showlegend' incorreto"

# Teste para verificar o campo 'showlegend'  esta incorreto
def test_data_showlegend02(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    showlegend = data_entry.get('showlegend')
    assert showlegend is not None, "Campo 'showlegend' não encontrado em data"
    assert showlegend is False, "Valor do campo 'showlegend' incorreto"

# Teste para verificar o campo 'values' 
def test_data_values(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    values = data_entry.get('values')
    assert values is not None, "Campo 'values' não encontrado em data"
    assert values == [50.0, 50.0, 50.0], "Valor do campo 'values' incorreto"

# Teste para verificar o campo 'type' 
def test_data_type(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    chart_type = data_entry.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado em data"
    assert chart_type == 'pie', "Valor do campo 'type' incorreto"

# Teste para verificar o campo 'type' esta incorreto
def test_data_type02(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    chart_type = data_entry.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado em data"
    assert chart_type == 'funnel', "Valor do campo 'type' incorreto"


#layout tests

# Teste para verificar o campo 'template' 
def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get('template')
    assert template is not None, "Campo 'template' não encontrado em layout"
    assert template == {'data': {'scatter': [{'type': 'scatter'}]}}, "Valor do campo 'template' incorreto"

# Teste para verificar o campo 'legend' 
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get('legend')
    assert legend is not None, "Campo 'legend' não encontrado em layout"
    assert legend == {'tracegroupgap': 0}, "Valor do campo 'legend' incorreto"

# Teste para verificar o campo 'margin' 
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get('margin')
    assert margin is not None, "Campo 'margin' não encontrado em layout"
    assert margin == {'t': 30, 'l': 30, 'r': 30, 'b': 30}, "Valor do campo 'margin' incorreto"

# Teste para verificar o campo 'extendpiecolors' 
def test_layout_extendpiecolors(generated_chart):
    data, layout = generated_chart
    extendpiecolors = layout.get('extendpiecolors')
    assert extendpiecolors is not None, "Campo 'extendpiecolors' não encontrado em layout"
    assert extendpiecolors is True, "Valor do campo 'extendpiecolors' incorreto"

# Teste para verificar o campo 'xaxis'
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get('xaxis')
    assert xaxis is not None, "Campo 'xaxis' não encontrado em layout"
    assert xaxis == {'categoryorder': 'trace'}, "Valor do campo 'xaxis' incorreto"
