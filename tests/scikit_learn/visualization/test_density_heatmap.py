from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# DensityHeatmap

import pdb;pdb.set_trace()

df = util.iris_polars()

@pytest.fixture
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'histogram2d',
            'display_legend': "AUTO",
            "x": [{
                "binning": "EQUAL_INTERVAL",
                "bins": 20,
                "binSize": 10,
                "emptyBins": "ZEROS",
                "decimal_places": 2,
                "group_others": True,
                "sorting": "NATURAL",
                "attribute": "petalwidth"
            }],
            "color_scale": [
                "#245668",
                "#0f7279",
                "#0d8f81",
                "#39ab7e",
                "#6ec574",
                "#a9dc67",
                "#edef5d"
            ],
            "y": [{
                "attribute": "petallength",
                "aggregation": "MIN",
                "displayOn": "left",
                "decimal_places": 2,
                "strokeSize": 0,
            }],
            "x_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "decimal_places": 2,
            },
            "y_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "decimal_places": 2,
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

# Definir uma escala de cores personalizada
custom_colors = plotly.colors.sequential.Viridis
# Gerar o gráfico com a escala de cores personalizada
fig = px.density_heatmap(df_pandas, x="sepallength", y="sepalwidth", color_continuous_scale=custom_colors, marginal_x="box", marginal_y="violin", title="test")

#fig = px.density_heatmap(df, x=df_select_result, y=df_select_result1, marginal_x="box", marginal_y="violin")

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']
'''

#test data
def test_type(generated_chart):
    data, layout = generated_chart
    assert any(item.get('type') == 'violin' for item in data), "Tipo data secundario'"

def test_chart_type_histogram2d(generated_chart):
    data, layout = generated_chart
    assert data[0].get('type') == 'histogram2d', "Tipo de gráfico correto para o primeiro objeto"

# Teste para verificar se o tipo do gráfico é 'histogram2d's
def test_chart_type(generated_chart):
    data, layout = generated_chart
    assert all(item.get('type') == 'box' for item in data), "Tipo de gráfico secundário"

# Teste para verificar se a legenda não está definida para nenhum item do gráfico
def test_chart_legend(generated_chart):
    data, layout = generated_chart
    assert all('showlegend' not in item for item in data), "Legenda definida para pelo menos um item do gráfico"

# Teste para verificar se as cores do marcador são definidas corretamente
def test_marker_colors(generated_chart):
    data, layout = generated_chart
    marker_colors = [item.get('marker', {}).get('color') for item in data]
    expected_colors = ['#245668'] * len(data)  
    assert marker_colors == expected_colors

# Teste para verificar se o eixo x está definido corretamente
def test_x_axis(generated_chart):
    data, layout = generated_chart
    x_axes = [item.get('xaxis') for item in data]
    expected_x_axes = ['x', 'x3', 'x2'] 
    assert x_axes == expected_x_axes

# Teste para verificar se o eixo y está definido corretamente
def test_y_axis(generated_chart):
    data, layout = generated_chart
    y_axes = [item.get('yaxis') for item in data]
    expected_y_axes = ['y', 'y3', 'y2']  
    assert y_axes == expected_y_axes

#testlayout

# Teste para verificar o campo 'xaxis' no layout
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get('xaxis')
    assert xaxis is not None, "Campo 'xaxis' não encontrado no layout"
    assert xaxis.get('title') == {'text': 'teste'}, "Valor do campo 'title' em 'xaxis' incorreto"
    assert xaxis.get('categoryorder') == 'trace', "Valor do campo 'categoryorder' em 'xaxis' incorreto"
    assert xaxis.get('categoryarray') == ['Dinner', 'Lunch'], "Valor do campo 'categoryarray' em 'xaxis' incorreto"

# Teste para verificar o campo 'yaxis' no layout
def test_layout_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis = layout.get('yaxis')
    assert yaxis is not None, "Campo 'yaxis' não encontrado no layout"
    assert yaxis.get('title') == {'text': 'min(total_bill)'}, "Valor do campo 'title' em 'yaxis' incorreto"

# Teste para verificar o campo 'legend' no layout
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get('legend')
    assert legend is not None, "Campo 'legend' não encontrado no layout"
    assert legend.get('title') == {'text': 'Legenda'}, "Valor do campo 'title' em 'legend' incorreto"
    assert legend.get('tracegroupgap') == 0, "Valor do campo 'tracegroupgap' em 'legend' incorreto"

# Teste para verificar o campo 'margin' no layout
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get('margin')
    assert margin is not None, "Campo 'margin' não encontrado no layout"
    assert margin == {'t': 30, 'l': 30, 'r': 30, 'b': 30}, "Valor do campo 'margin' incorreto"

# Teste para verificar o campo 'boxmode' no layout
def test_layout_boxmode(generated_chart):
    data, layout = generated_chart
    boxmode = layout.get('boxmode')
    assert boxmode == 'overlay', "Valor do campo 'boxmode' incorreto"

# Teste para verificar o campo 'showlegend' no layout
def test_layout_showlegend(generated_chart):
    data, layout = generated_chart
    showlegend = layout.get('showlegend')
    assert showlegend is not None, "Campo 'showlegend' não encontrado no layout"
    assert showlegend is True, "Valor do campo 'showlegend' incorreto"
