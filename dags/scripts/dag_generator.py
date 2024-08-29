import requests
from jinja2 import Environment, FileSystemLoader
#from juicer.jobs import code_gen 

import yaml
import os
#from juicer.jobs.code_gen import _generate 
from juicer.jobs.code_gen import generate

from gettext import translation
from io import StringIO
'''
def convert_lemonade_to_python(workflow_id):
    config_path = 'conf/juicer-config.yaml'
    with open(config_path) as config_file:
        juicer_config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    
    vars_path = None  
    custom_vars = None
    if vars_path:
        with open(vars_path) as vars_file:
            custom_vars = yaml.load(vars_file.read(), Loader=yaml.FullLoader)
    
    locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
    lang = 'pt'  
    t = translation('messages', locales_path, [lang], fallback=True)
    t.install()
    
    job_id = 0  
    execute_main = False  
    deploy = False  
    export_notebook = False  
    plain = False  
    from_meta = False  
    variant = None  
    json_file = None  
    
    out = StringIO()
    #retornar meta e n spark , vizualizar a função generate
    try:
        _generate(
            workflow_id=workflow_id,
            job_id=job_id,
            execute_main=execute_main,
            params={"plain": plain},
            config=juicer_config,
            deploy=deploy,
            export_notebook=export_notebook,
            plain=plain,
            custom_vars=custom_vars,
            lang=lang,
            from_meta=from_meta,
            variant=variant,
            json_file=json_file,
            out=out
        )
        
        generated_code = out.getvalue()

        current_directory = os.getcwd()
        output_directory = os.path.join(current_directory, 'workflows')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_file_path = os.path.join(output_directory, f'{workflow_id}.py')

        with open(output_file_path, 'w') as code_file:
            code_file.write(generated_code)

        print(f"Código salvo em: {output_file_path}")

        return output_file_path
    
    except Exception as e:
        raise
        #raise Exception(f"Erro ao gerar código: {str(e)}")
'''
def convert_lemonade_to_python(workflow_id):
    config_path = 'conf/juicer-config.yaml'
    os.environ['JUICER_CONF'] = config_path
    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    template_name = 'python code'
    lang = 'pt'
    result = generate(workflow_id, template_name, config, lang)
    if result['code'] == '':
        print("An error occurred while generating the code")
    else:
        current_directory = os.getcwd()
        output_directory = os.path.join(current_directory, 'workflows')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
        output_file_path = os.path.join(output_directory, f'workflow_{workflow_id}.py')
        with open(output_file_path, 'w') as code_file:
            code_file.write(result['code'])

            print(f"Code saved in: {output_file_path}")

        return output_file_path
    

def generate_dag(pipeline_id, pipeline_description, workflows):

    current_working_dir = os.getcwd()
    templates_dir = os.path.join(current_working_dir, 'templates')

    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('dag_template.tmpl')
    
    rendered_code = template.render(
        pipeline_id=pipeline_id,
        pipeline_description=pipeline_description,
        workflows=workflows
    )

    dag_file_path = f"airflow/dags/dag_{pipeline_id}.py"
    
    with open(dag_file_path, "w") as f:
        f.write(rendered_code)

    print(f"DAG gerada e salva em : {dag_file_path}")


def fetch_pipeline_from_api(pipeline_api_url, headers=None):
    response = requests.get(pipeline_api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Erro ao acessar API: {response.status_code}")

pipeline_api_url = 'https://dev.lemonade.org.br/api/v1/tahiti/pipelines/36'

headers = {
    'x-auth-token': '123456',
}

pipeline_data = fetch_pipeline_from_api(pipeline_api_url, headers=headers)
#print(pipeline_data)
#python_code_path = convert_lemonade_to_python(742)
#workflow_id = 813  
#workflow_id = 742  

#codigo_python = convert_lemonade_to_python02(workflow_id)
#print(codigo_python)

if pipeline_data['status'] == 'OK' and pipeline_data['data']:
    pipeline = pipeline_data['data'][0]
    pipeline_id = pipeline['id']
    pipeline_description = pipeline['description']
    #verificar mais informações que podem ser incorporadas da API
    workflows = []
    #converter cada workflow em código Python
    #python_code_path = convert_lemonade_to_python(workflow_id) #teste
    
    #workflows.append({'id': workflow_id, 'python_file_path': python_code_path})
    
    for step in pipeline['steps']:
        workflow_id = step['workflow']['id']
        #print(workflow_id)
        python_code_path = convert_lemonade_to_python(workflow_id)
        workflows.append({'id': workflow_id, 'python_file_path': python_code_path})
    
    generate_dag(pipeline_id, pipeline_description, workflows)#gera a DAG  com o jinja2
else:
    print("Nenhuma pipeline válida encontrada na resposta da API.")

#dag_generator.py
