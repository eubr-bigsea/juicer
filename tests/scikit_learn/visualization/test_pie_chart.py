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
            "attribute": "species"
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
#trechos do dicionario do codigo gerado
dict0_chart = data[0]
dict1_chart = data[1]
dict2_chart = data[1]

color_chart = data[2]['marker']['color']
type_chart = data[0]['type']
showlegend_chart = dict1_chart['showlegend']
print(showlegend_chart)

print(type_chart)

## Rever o código ##
# Codigo de teste
'''
df_select = df.select("petalwidth")
df_select_result = df_select.lazy().select("petalwidth").collect()
df_select1 = df.select("petallength")
df_select_result1 = df_select1.lazy().select("petallength").collect()
df_select_result_pd = df_select_result.to_pandas()
df_select_result1_pd = df_select_result1.to_pandas()
   '''
df_pol = df.collect()
df_pandas = df_pol.to_pandas()

# Definir uma escala de cores personalizada
custom_colors = plotly.colors.sequential.Viridis
# Gerar o gráfico com a escala de cores personalizada
#verificar
valor = 1 
fig = px.pie(df, values=valor, names='species', color_discrete_sequence=px.colors.sequential.RdBu)

#fig = px.density_heatmap(df, x=df_select_result, y=df_select_result1, marginal_x="box", marginal_y="violin")

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
'''def test_test_dentity_heatmap_type():
    assert type_chart == type_test

#teste legenda
def test_test_dentity_heatmap_legend():
    assert showlegend_chart == showlegend_test 
    
#teste escala de cores
def test_test_dentity_heatmap_colorscale():
    assert color_chart == color_test
'''
#layout tests
   
