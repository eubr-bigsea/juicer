from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors

# DensityHeatmap

#def test_test_dentity_heatmap():
    
df = util.iris_polars()
arguments = {
    'parameters': {
        'type': 'bar',
        'display_legend': "AUTO",
        "x": [{
            "binning": None,
            "bins": 20,
            "binSize": 10,
            "emptyBins": "ZEROS",
            "decimal_places": 2,
            "group_others": True,
            "sorting": "NATURAL",
            "attribute": "class",
            
        }],
        "palette": [
            "#245668",
            "#0f7279",
            "#0d8f81",
            "#39ab7e",
            "#6ec574",
            "#a9dc67",
            "#edef5d"
        ],
        "y": [{
            "attribute": "sepallength",
            "aggregation": "AVG",
            "displayOn": "left",
            "decimal_places": 2,
            "strokeSize": 0,
            "enabled": True
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
        'subgraph_orientation': "v",
        
        "task_id": "1"
    },
    'named_inputs': {
        'input data': "iris",
    },
    'named_outputs': {
        'output data': 'out'
    }
}


    

instance = VisualizationOperation(**arguments)
print(instance.generate_code())
def emit_event(*args, **kwargs):
    print(args, kwargs)

vis_globals = dict(iris=df, emit_event=emit_event)
code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
result = util.execute(code, 
                          vis_globals)

# Use result.get('d') to get the Python dict containing the chart
generated_chart = result.get('d')

data = generated_chart['data']
layout = generated_chart['layout']