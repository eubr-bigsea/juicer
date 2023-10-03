from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# histogram2dcontour

import pdb;pdb.set_trace()
    
df = util.iris2_polars()

@pytest.fixture
def get_df():
    return util.iris2_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'histogram2dcontour',
        'display_legend': "HIDE",
        "x": [{
            "binning": "EQUAL_INTERVAL",
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
            "attribute": "PetalLengthCm"
        }],
        "color_scale": [
            "#0000ff",
            "#ff0000"
        ],
        "y": [{
            "attribute": "PetalWidthCm",
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

fig = px.density_contour(df_pandas, x=df_pandas['petallength'], y=df_pandas['petalwidth'])
fig.update_traces(contours_coloring="fill", contours_showlabels = True)

# Converter em JSONc
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']
'''

#data tests

# Teste para verificar o campo 'contours' 
def test_data_contours(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    contours = histogram2dcontour_data.get('contours')
    assert contours is not None, "Campo 'contours' não encontrado no objeto de dados"
    expected_contours = {'coloring': 'fill', 'showlabels': True}
    assert contours == expected_contours, "Valores do campo 'contours' incorretos"

# Teste para verificar o campo 'hovertemplate' 
def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    hovertemplate = histogram2dcontour_data.get('hovertemplate')
    assert hovertemplate is not None, "Campo 'hovertemplate' não encontrado no objeto de dados"
    

# Teste para verificar o campo 'legendgroup' 
def test_data_legendgroup(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    legendgroup = histogram2dcontour_data.get('legendgroup')
    assert legendgroup is not None, "Campo 'legendgroup' não encontrado no objeto de dados"
    assert legendgroup == '', "Valor do campo 'legendgroup' incorreto"


# Teste para verificar o campo 'line' 
def test_data_line(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    line = histogram2dcontour_data.get('line')
    assert line is not None, "Campo 'line' não encontrado no objeto de dados"
    

# Teste para verificar o campo 'name' 
def test_data_name(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    name = histogram2dcontour_data.get('name')
    assert name is not None, "Campo 'name' não encontrado no objeto de dados"
    assert name == '', "Valor do campo 'name' incorreto"


# Teste para verificar o campo 'showlegend' 
def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0] 
    showlegend = histogram2dcontour_data.get('showlegend')
    assert showlegend is not None, "Campo 'showlegend' não encontrado no objeto de dados"
    assert showlegend == False, "Valor do campo 'showlegend' incorreto"


# Teste para verificar o campo 'x' 
def test_data_x(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    x = histogram2dcontour_data.get('x')
    assert x is not None, "Campo 'x' não encontrado no objeto de dados"
    expected_x = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert x == expected_x, "Valores do campo 'x' incorretos"

# Teste para verificar o campo 'y' 
def test_data_y(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    y = histogram2dcontour_data.get('y')
    assert y is not None, "Campo 'y' não encontrado no objeto de dados"
    expected_y = [0.1, 0.1, 0.2, 0.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.9, 1.4, 1.6, 1.8, 2.0, 2.0]
    assert y == expected_y, "Valores do campo 'y' incorretos"

# Teste para verificar o campo 'type' 
def test_data_type(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]  
    chart_type = histogram2dcontour_data.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado no objeto de dados"
    assert chart_type == 'histogram2dcontour', "Valor do campo 'type' incorreto"


# Teste para verificar o campo 'colorscale' 
def test_data_colorscale(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0] 
    colorscale = histogram2dcontour_data.get('colorscale')
    assert colorscale is not None, "Campo 'colorscale' não encontrado no objeto de dados"
    
#layout tests

# Teste para verificar o campo 'template' 
def test_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    assert layout_template is not None, "Campo 'template' não encontrado no objeto de layout"
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}
    assert layout_template == expected_template, "Valores do campo 'template' incorretos"

# Teste para verificar o campo 'xaxis' 
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    assert layout_xaxis is not None, "Campo 'xaxis' não encontrado no objeto de layout"
    expected_xaxis = {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'PetalLengthCm'}, 'categoryorder': 'trace'}
    assert layout_xaxis == expected_xaxis, "Valores do campo 'xaxis' incorretos"

# Teste para verificar o campo 'yaxis' 
def test_layout_yaxis(generated_chart):
    data, layout = generated_chart
    layout_yaxis = layout.get('yaxis')
    assert layout_yaxis is not None, "Campo 'yaxis' não encontrado no objeto de layout"
    expected_yaxis = {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'min(PetalWidthCm)'}}
    assert layout_yaxis == expected_yaxis, "Valores do campo 'yaxis' incorretos"

# Teste para verificar o campo 'legend' 
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get('legend')
    assert layout_legend is not None, "Campo 'legend' não encontrado no objeto de layout"
    expected_legend = {'tracegroupgap': 0}
    assert layout_legend == expected_legend, "Valores do campo 'legend' incorretos"

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
    assert layout_showlegend == False, "Valor do campo 'showlegend' incorreto"


