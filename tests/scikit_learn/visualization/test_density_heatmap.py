#from tests.scikit_learn import util
from tests.scikit_learn import util

#from tests.scikit_learn.util import *
#from juicer.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px


# DensityHeatmap

def test_test_dentity_heatmap():
    
    df = util.iris_polars()
    arguments = {
            'parameters': {
                'type':'histogram2d', 
                'display_legend':"AUTO", 
                "x":[
                    {
                        "binning": "EQUAL_INTERVAL",
                        "bins": 20,
                        "binSize": 10,
                        "emptyBins": "ZEROS",
                        "decimal_places": 2,
                        "group_others": True,
                        "sorting": "NATURAL",
                        "attribute": "petalwidth"
                        }
                    ], "color_scale":[
                        "#245668",
                        "#0f7279",
                        "#0d8f81",
                        "#39ab7e",
                        "#6ec574",
                        "#a9dc67",
                        "#edef5d"
                        ],
                    "y":[{"attribute": "petallength",
                        "aggregation": "MIN",
                        "displayOn": "left",
                        "decimal_places": 2,
                        "strokeSize": 0,
                        }],
                    "x_axis":{
                        "logScale": False,
                        "display": True,
                        "displayLabel": True,
                        "decimal_places": 2,
                        }, "y_axis":{
                            "logScale": False,
                            "display": True,
                            "displayLabel": True,
                            "decimal_places": 2,
                            }, "task_id":"0"},
                        'named_inputs': {
                            'input data': "iris",
                            },
                        'named_outputs': {
                            'output data': 'out'
                            }
                        }

    instance = VisualizationOperation(**arguments)
    # Gerar codigo de todo o template ??
    # Como expecificar a execução de um grafico expecifico no template ??
    # JSON ??

    # Gerar código do template
    
    '''instance.template = {%- if plot_code == 'heatmap' %}
    fig = px.density_heatmap(pandas_df, 
                                 x='dim_0', 
                                 y='aggr_0', 
                                 marginal_x="box", 
                                 marginal_y="violin",
                                 title='{{op.title}}',
                                 labels=labels,
                                 color_continuous_scale={{op.color_scale}})
    {% endif %}'''

    '''
    
    # Definir o código específico para o heatmap
    instance.template_vars['plot_code'] = 'heatmap'
    
    # Gerar código a partir do template
    code = instance.generate_code()
    
    # Extrair o trecho do código do heatmap
    heatmap_code = instance.template.extract_part(code, 'fig =', '})')
    '''
    def emit_event(*args, **kwargs):
        print(args, kwargs)

    vis_globals = dict(iris=df, emit_event=emit_event)
    code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
    result = util.execute(code, 
                          vis_globals)

    # Use result.get('d') to get the Python dict containing the chart
    generated_chart = result.get('d')
    import pdb;pdb.set_trace()

    #print(result.keys())
    atributos = result.keys()
    #print(result['out'])
    #print(type(generated_chart))
    #chaves2 = generated_chart.keys()
    #print(chaves2)
    #print(type(result))
    #chaves = result.keys()
    #print(chaves)
    #print(result.items)
    atributo = result.get('d')
    print(atributo)  # Saída: valor2
    print(type(atributo))
    print(atributo.keys())
    data = atributo['data']
    layout = atributo['layout']
    print(data)
    coloraxis = data[0]['y']
    print(coloraxis)
    print(layout)
    scatter = layout.keys()
    print(scatter)
    scatter2 = layout['template']['data']
    print(scatter2)
    ## Rever o código ##
    # Codigo de teste
    df_select = df.select("sepallength")
    df_select_result = df_select.lazy().select("sepallength").collect()
    df_select1 = df.select("sepalwidth")
    df_select_result1 = df_select1.lazy().select("sepalwidth").collect()
    print(df_select_result)
    print(df_select_result1)
    df_select_result_pd = df_select_result.to_pandas()
    df_select_result1_pd = df_select_result1.to_pandas()
    #print(df_select)
   #fig = px.density_heatmap(df, x=df_select_result, y=df_select_result1, marginal_x="box", marginal_y="violin")
    #df_pandas = df.to_pandas()
    df_pol = df.collect()
    df_pandas = df_pol.to_pandas()
    fig = px.density_heatmap(df_pandas, x="sepallength", y="sepalwidth", marginal_x="box", marginal_y="violin")

    #fig = px.density_heatmap(df, x=df_select_result, y=df_select_result1, marginal_x="box", marginal_y="violin")


    # Converter em JSON
    fig_json = fig.to_json()
    dicionario = json.loads(fig_json)
    print(dicionario)
    print(dicionario.keys())
    print(type(fig_json))
   
   
    '''
    # Tentar carregar o JSON como um objeto Python
    try:
        json_obj = json.loads(fig_json)
    except json.JSONDecodeError:
        assert False, "Falha ao carregar o JSON gerado"

    # Verificar se o JSON carregado possui as informações esperadas
    assert "data" in json_obj
    assert "layout" in json_obj
    
    # Verificar se o JSON gerado é igual ao JSON da vizualização Lemonade
    # teste_out
    # assert result['out'].equals(test_df)

    assert fig_json == fig.to_json()

'''
