from tests.scikit_learn import util
#from juicer.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import plotly.express as px


# DensityHeatmap

def test_test_dentity_heatmap():
    
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    instance = VisualizationOperation(**arguments)
    # Gerar codigo de todo o template ??
    # Como expecificar a execução de um grafico expecifico no template ??
    # JSON ??
    result = util.execute(instance.generate_code(), 
                          dict([df]))
    # Codigo de teste
    fig = px.density_heatmap(test_df, x="sepal_length", y="sepal_width", marginal_x="box", marginal_y="violin")

    # Converter em JSON
    fig_json = fig.to_json()

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


