from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# sunburst
    
df = util.titanic_polars()

arguments = {
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
        'input data': "titanic",
    },
    'named_outputs': {
        'output data': 'out'
    }
}


    

instance = VisualizationOperation(**arguments)
    
def emit_event(*args, **kwargs):
    print(args, kwargs)

vis_globals = dict(titanic=df, emit_event=emit_event)
code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
result = util.execute(code, 
                          vis_globals)

# Use result.get('d') to get the Python dict containing the chart
generated_chart = result.get('d')
import pdb;pdb.set_trace()

data = generated_chart['data']
layout = generated_chart['layout']

print(data)
print(layout)
dff = pd.DataFrame(data)

valores = dff.count()
fig = px.sunburst(df, path="pclass", values=valores, color='day')

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)

data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

print(data1)
#trechos do dicionario do codigo gerado
dict0_test = data1[0]
dict1_test = data1[1]
dict2_test = data1[1]
print(dict0_test)
print(dict1_test)
print(dict2_test)



    
color_test = data1[2]['marker']['color']
type_test = data1[0]['type']
showlegend_test = dict1_test['showlegend']
print(showlegend_test)
print(type_test)

#data tests    
#teste type
def test_dentity_heatmap_type():
    assert type_chart == type_test

#teste legenda
def test_dentity_heatmap_legend():
    assert showlegend_chart == showlegend_test 
    
#teste escala de cores
def test_dentity_heatmap_colorscale():
    assert color_chart == color_test

#layout tests
   