from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors

#testes Comuns
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
    generated_chart_code = result.get('d')
    data = generated_chart_code['data']
    layout = generated_chart_code['layout']
    print(data)
    return data,layout

# Teste para verificar o campo 'type' 
def test_data_type(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]  
    chart_type = sunburst_data.get('type')
    assert chart_type is not None, "Campo 'type' não encontrado no objeto de dados"
    assert chart_type == 'sunburst', "Valor do campo 'type' incorreto"

# Teste para verificar o campo 'margin' 
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    layout_margin = layout.get('margin')
    assert layout_margin is not None, "Campo 'margin' não encontrado no objeto de layout"
    expected_margin = {'t': 30, 'l': 30, 'r': 30, 'b': 30}
    assert layout_margin == expected_margin, "Valores do campo 'margin' incorretos"