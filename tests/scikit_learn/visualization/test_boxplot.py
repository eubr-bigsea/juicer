from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors

# Boxplot

import pdb;pdb.set_trace()
    
df = util.tips_polars()

@pytest.fixture
def get_df():
    return util.tips_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'boxplot',
        'display_legend': "RIGHT",
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
            "attribute": "time",
            "displayLabel": "teste"
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
            "attribute": "total_bill",
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
#
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

fig = px.box(df_pandas, x=df_pandas['time'], y=df_pandas['total_bill'], points="all", color=df_pandas['smoker'])


# Converter em JSONc
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

'''


#data tests

# Teste para verificar se o tipo do gráfico é 'box'
def test_boxplot_type(generated_chart):
    data, layout = generated_chart
    assert all(item.get('type') == 'box' for item in data), "Tipo de gráfico incorreto"


# Teste para verificar o campo 'boxpoints' 
def test_boxplot_boxpoints(generated_chart):
    data, layout = generated_chart
    boxpoints = [item.get('boxpoints') for item in data]
    expected_boxpoints = ['all', 'all']  
    assert boxpoints == expected_boxpoints, "Valores do campo 'boxpoints' incorretos"

# Teste para verificar o campo 'legendgroup' 
def test_boxplot_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroup = [item.get('legendgroup') for item in data]
    expected_legendgroup = ['Dinner', 'Lunch'] 
    assert legendgroup == expected_legendgroup, "Valores do campo 'legendgroup' incorretos"

# Teste para verificar o campo 'showlegend' 
def test_layout_showlegend(generated_chart):
    data, layout = generated_chart
    layout_showlegend = layout.get('showlegend')
    expected_layout_showlegend = True  
    assert layout_showlegend == expected_layout_showlegend, "Valor do campo 'showlegend' no layout incorreto"


# Teste para verificar o campo 'alignmentgroup' 
def test_boxplot_alignmentgroup(generated_chart):
    data, layout = generated_chart
    alignmentgroup = [item.get('alignmentgroup') for item in data]
    expected_alignmentgroup = ['True', 'True']
    assert alignmentgroup == expected_alignmentgroup, "Valores do campo 'alignmentgroup' incorretos"

# Teste para verificar o campo 'hovertemplate' 
def test_boxplot_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplate = [item.get('hovertemplate') for item in data]
    expected_hovertemplate = ['teste=%{x}<br>min(total_bill)=%{y}<extra></extra>', 'teste=%{x}<br>min(total_bill)=%{y}<extra></extra>']
    assert hovertemplate == expected_hovertemplate, "Valores do campo 'hovertemplate' incorretos"

# Teste para verificar o campo 'marker' 
def test_boxplot_marker(generated_chart):
    data, layout = generated_chart
    markers = [item.get('marker') for item in data]
    expected_markers = [
        {'color': '#636efa'}, {'color': '#EF553B'}
    ]
    assert markers == expected_markers, "Valores do campo 'marker' incorretos"

# Teste para verificar o campo 'name' 
def test_boxplot_name(generated_chart):
    data, layout = generated_chart
    names = [item.get('name') for item in data]
    expected_names = ['Dinner', 'Lunch']
    assert names == expected_names, "Valores do campo 'name' incorretos"

# Teste para verificar o campo 'notched' 
def test_boxplot_notched(generated_chart):
    data, layout = generated_chart
    notched_values = [item.get('notched') for item in data]
    expected_notched_values = [False, False]
    assert notched_values == expected_notched_values, "Valores do campo 'notched' incorretos"

# Teste para verificar o campo 'offsetgroup' 
def test_boxplot_offsetgroup(generated_chart):
    data, layout = generated_chart
    offsetgroups = [item.get('offsetgroup') for item in data]
    expected_offsetgroups = ['Dinner', 'Lunch']
    assert offsetgroups == expected_offsetgroups, "Valores do campo 'offsetgroup' incorretos"

# Teste para verificar o campo 'orientation' 
def test_boxplot_orientation(generated_chart):
    data, layout = generated_chart
    orientations = [item.get('orientation') for item in data]
    expected_orientations = ['v', 'v']
    assert orientations == expected_orientations, "Valores do campo 'orientation' incorretos"

# Teste para verificar o campo 'x' 
def test_boxplot_x(generated_chart):
    data, layout = generated_chart
    x_values = [item.get('x') for item in data]
    expected_x_values = [['Dinner'], ['Lunch']]
    assert x_values == expected_x_values, "Valores do campo 'x' incorretos"

# Teste para verificar o campo 'x0' 
def test_boxplot_x0(generated_chart):
    data, layout = generated_chart
    x0_values = [item.get('x0') for item in data]
    expected_x0_values = [' ', ' ']
    assert x0_values == expected_x0_values, "Valores do campo 'x0' incorretos"

# Teste para verificar o campo 'xaxis' 
def test_boxplot_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis_values = [item.get('xaxis') for item in data]
    expected_xaxis_values = ['x', 'x']
    assert xaxis_values == expected_xaxis_values, "Valores do campo 'xaxis' incorretos"

# Teste para verificar o campo 'y' 
def test_boxplot_y(generated_chart):
    data, layout = generated_chart
    y_values = [item.get('y') for item in data]
    expected_y_values = [[8.77], [9.55]]
    assert y_values == expected_y_values, "Valores do campo 'y' incorretos"

# Teste para verificar o campo 'y0' 
def test_boxplot_y0(generated_chart):
    data, layout = generated_chart
    y0_values = [item.get('y0') for item in data]
    expected_y0_values = [' ', ' ']
    assert y0_values == expected_y0_values, "Valores do campo 'y0' incorretos"

# Teste para verificar o campo 'yaxis' 
def test_boxplot_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis_values = [item.get('yaxis') for item in data]
    expected_yaxis_values = ['y', 'y']
    assert yaxis_values == expected_yaxis_values, "Valores do campo 'yaxis' incorretos"

# Teste para verificar o campo 'quartilemethod' 
def test_boxplot_quartilemethod(generated_chart):
    data, layout = generated_chart
    quartilemethods = [item.get('quartilemethod') for item in data]
    expected_quartilemethods = ['exclusive', 'exclusive']
    assert quartilemethods == expected_quartilemethods, "Valores do campo 'quartilemethod' incorretos"


#layout tests 

# Teste para verificar se o template do layout está definido corretamente
def test_funnel_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}  
    assert layout_template == expected_template, "Template do layout incorreto"

# Teste para verificar as configurações do eixo x (xaxis)
def test_funnel_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    expected_xaxis = {
        'anchor': 'y',
        'domain': [0.0, 1.0],
        'title': {'text': 'teste'},
        'categoryorder': 'trace',
        'categoryarray': ['Dinner', 'Lunch']
    }  
    assert layout_xaxis == expected_xaxis, "Configurações do eixo x (xaxis) do layout incorretas"

# Teste para verificar as configurações do eixo y (yaxis) 
def test_funnel_layout_yaxis(generated_chart):
    data, layout = generated_chart
    layout_yaxis = layout.get('yaxis')
    expected_yaxis = {
        'anchor': 'x',
        'domain': [0.0, 1.0],
        'title': {'text': 'min(total_bill)'}
    }  
    assert layout_yaxis == expected_yaxis, "Configurações do eixo y (yaxis) do layout incorretas"

# Teste para verificar as configurações da legenda (legend) 
def test_funnel_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get('legend')
    expected_legend = {
        'title': {'text': 'Legenda'},
        'tracegroupgap': 0,
        'orientation': 'v',
        'yanchor': 'top',
        'y': 0.99,
        'xanchor': 'right',
        'x': 0.99
    }  
    assert layout_legend == expected_legend, "Configurações da legenda (legend) do layout incorretas"

# Teste para verificar as configurações das margens (margin) no layout
def test_funnel_layout_margin(generated_chart):
    data, layout = generated_chart
    layout_margin = layout.get('margin')
    expected_margin = {'t': 30, 'l': 30, 'r': 30, 'b': 30}  
    assert layout_margin == expected_margin, "Configurações das margens (margin) do layout incorretas"

# Teste para verificar se a configuração de 'boxmode' no layout está definida corretamente
def test_funnel_layout_boxmode(generated_chart):
    data, layout = generated_chart
    layout_boxmode = layout.get('boxmode')
    expected_boxmode = 'overlay'  
    assert layout_boxmode == expected_boxmode, "Configuração de 'boxmode' do layout incorreta"

# Teste para verificar a configuração 'showlegend' no layout 
def test_funnel_layout_showlegend(generated_chart):
    data, layout = generated_chart
    layout_showlegend = layout.get('showlegend')
    expected_showlegend = True  
    assert layout_showlegend == expected_showlegend, "Configuração 'showlegend' do layout incorreta"




