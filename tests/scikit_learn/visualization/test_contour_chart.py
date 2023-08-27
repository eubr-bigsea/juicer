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
    
df = util.iris_polars()

arguments = {
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
            "attribute": "petallength"
        }],
        "color_scale": [
            "#0000ff",
            "#ff0000"
        ],
        "y": [{
            "attribute": "petalwidth",
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


    

instance = VisualizationOperation(**arguments)
    
def emit_event(*args, **kwargs):
    print(args, kwargs)

vis_globals = dict(iris=df, emit_event=emit_event)
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

type_chart = data[0]['type']
showlegend_chart = data[0]['showlegend']
colorscale_chart = data[0]['colorscale']
x_chart = data[0]['x']
y_chart = data[0]['y']
print(type_chart)
print(showlegend_chart)
print(colorscale_chart)
print(x_chart)
print(y_chart)


df_pol = df.collect()
df_pandas = df_pol.to_pandas()

fig = px.density_contour(df_pandas, x=df_pandas['petallength'], y=df_pandas['petalwidth'])
fig.update_traces(contours_coloring="fill", contours_showlabels = True)

# Converter em JSONc
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

print(data1)
print(layout1)

type_test = data1[0]['type']
showlegend_test = data1[0]['showlegend']
'''colorscale_test = data1[0]['colorscale']
x_test = data1[0]['x']
y_test = data1[0]['y']'''

print(type_test)
print(showlegend_test)



#data tests    
#teste type

def test_boxplot_type():
    assert type_chart == type_test

'''#teste eixos
def test_eixo_x():
    assert x_chart == x_test

def test_eixo_y():
    assert y_chart == y_test'''

#legenda
def test_boxplot_showlegend():
    assert showlegend_chart == showlegend_test


