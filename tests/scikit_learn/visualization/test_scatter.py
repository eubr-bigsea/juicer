from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors

# Scatter

df = util.iris_polars()
arguments = {
    'parameters': {
        'type': 'scatter',
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
        "color_attribute": {
            "id": 5261,
            "name": "class",
            "type": "CHARACTER",
            "size": 15,
            "nullable": False,
            "enumeration": False,
            "feature": False,
            "label": False,
            "key": False,
            "attribute": "class",
            "numeric": False,
            "integerType": False
        },
        "size_attribute": {
             "id": 5259,
             "name": "petallength",
             "type": "DECIMAL",
             "precision": 2,
             "scale": 1,
             "nullable": False,
             "enumeration": False,
             "feature": False,
             "label": False,
             "key": False,
             "attribute": "petallength",
             "numeric": True,
             "integerType": False
        },
        "y": [{
            "attribute": "petallength",
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
code ='\n'.join( ["import plotly.express as px","import json",instance.generate_code(),])
result = util.execute(code, 
                          vis_globals)

generated_chart = result.get('d')

data = generated_chart['data']
layout = generated_chart['layout']