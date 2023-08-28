from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# Grafico de pizza

#def test_test_dentity_heatmap():
    
df = util.iris_polars()

arguments = {
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


#color_chart = data[2]['marker']['color']
type_chart = data[0]['type']
showlegend_chart = data[0]['showlegend']
values_chart = data[0]['values']
print(showlegend_chart)

print(type_chart)
print(values_chart)


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

#



#   
type_test = data1[0]['type']
showlegend_test = data1[0]['showlegend']
values_test = data1[0]['values']

print(data1)
print(showlegend_test)
print(type_test)
print(values_test)

#data tests    
#teste type
def test_pie_type():
    assert type_chart == type_test

#teste legenda
def test_pie_legend():
    assert showlegend_chart == showlegend_test 
    
#teste escala de cores
def test_pie_values():
    assert values_chart == values_test

#layout tests
   