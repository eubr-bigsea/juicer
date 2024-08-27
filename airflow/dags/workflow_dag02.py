from jinja2 import Environment, FileSystemLoader
from juicer.jobs import code_gen #olhar o diretorio correto

import os

'''
Script para gerar as DAGS
'''

'''
def convert_lemonade_to_python(workflow_id):
    return f"~/Documentos/lemonade/docker-lemonade/juicer/workflows/code_{workflow_id}.py"
'''
def convert_lemonade_to_python(workflow_id):

    template_name = "python" 
    lang = "pt"  
    result = code_gen.generate(workflow_id, template_name, lang)
    
    if result['status'] == 'OK':
        return result['code']
    else:
        raise Exception(f"Erro ao gerar código: {result['message']}")
    
def generate_dag(pipeline_id, pipeline_description, workflows):

    env = Environment(loader=FileSystemLoader('/home/luiz/Documentos/lemonade/docker-lemonade/juicer/templates'))
    template = env.get_template('dag_template.tmpl')

    rendered_code = template.render( #renderiza e roda o template
        pipeline_id=pipeline_id,
        pipeline_description=pipeline_description,
        workflows=workflows
    )

    # Caminho onde será salvo o código Python da DAG
    dag_file_path = f"dag_{pipeline_id}.py"
    
    with open(dag_file_path, "w") as f:
        f.write(rendered_code)

    print(f"DAG generated and saved to: {dag_file_path}")

pipeline_id = 'pipelineGenerica01'
pipeline_description = 'Pipeline teste, sequencia de workflows lemonade para alguma aplicação'
#etapas da pipeline, exemplo
workflows = [
    {'id': '742', 'python_file_path': convert_lemonade_to_python('742')},
    {'id': '743', 'python_file_path': convert_lemonade_to_python('743')},
    {'id': '744', 'python_file_path': convert_lemonade_to_python('744')},
]

generate_dag(pipeline_id, pipeline_description, workflows)#Gera a DAG dinamicamente

