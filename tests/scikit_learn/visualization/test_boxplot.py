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
    
df = util.tips_polars()

arguments = {
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

boxpoints_chart = data[0]['boxpoints']
alignmentgroup_chart = data[0]['alignmentgroup']
legendgroup_chart = data[0]['legendgroup']
showlegend_chart = data[0]['showlegend']
type_chart = data[0]['type']

print(boxpoints_chart)
print(alignmentgroup_chart)
print(legendgroup_chart)
print(showlegend_chart)
print(type_chart)

df_pol = df.collect()
df_pandas = df_pol.to_pandas()

fig = px.box(df_pandas, x=df_pandas['time'], y=df_pandas['total_bill'], points="all", color=df_pandas['smoker'])


# Converter em JSONc
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

print(data1)
print(layout1)

boxpoints_test = data1[0]['boxpoints']
alignmentgroup_test = data1[0]['alignmentgroup']
legendgroup_test = data1[0]['legendgroup']
showlegend_test = data1[0]['showlegend']
type_test = data1[0]['type']

print(boxpoints_test)
print(alignmentgroup_test)
print(legendgroup_test)
print(showlegend_test)
print(type_test)


#data tests    
#teste type

def test_boxplot_type():
    assert type_chart == type_test

#teste alignmentgroup
def test_boxplot_alignmentgroup():
    assert alignmentgroup_chart == alignmentgroup_test

#teste legendgroup
def test_boxplot_legendgroup():
    assert legendgroup_chart == legendgroup_test

#pontos
def test_boxplot_boxpoints():
    assert boxpoints_chart == boxpoints_test

#legenda
def test_boxplot_showlegend():
    assert showlegend_chart == showlegend_test


