import requests
from jinja2 import Environment, FileSystemLoader
#from juicer.jobs import code_gen 

import yaml
import os
#from juicer.jobs.code_gen import _generate 
from juicer.jobs.code_gen import generate

from gettext import translation
from io import StringIO

import argparse
from datetime import datetime


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
'''
def convert_lemonade_to_python(workflow_id,config_path,lang,output_workflow_dir):

    config_path = 'conf/juicer-config.yaml'
    os.environ['JUICER_CONF'] = config_path

    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    template_name = 'python code'
    lang = lang
    result = generate(workflow_id, template_name, config, lang)

    if result['code'] == '':
        print("An error occurred while generating the code")
        return None
    else:
        output_file_path = os.path.join(output_workflow_dir, f'workflow_{workflow_id}.py')
        if not os.path.exists(output_workflow_dir):
            os.makedirs(output_workflow_dir)

        with open(output_file_path, 'w') as code_file:
            code_file.write(result['code'])
        print(f"Code saved in: {output_file_path}")
        return output_file_path    

'''
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

    dag_file_path = f"dags/dag_{pipeline_id}.py"
    
    with open(dag_file_path, "w") as f:
        f.write(rendered_code)

    print(f"DAG gerada e salva em : {dag_file_path}")
'''
def generate_dag(pipeline_id, pipeline_description, workflows, pipeline_user_name, pipeline_user_login, output_dags_dir, start_date):
    current_working_dir = os.getcwd()
    templates_dir = os.path.join(current_working_dir, 'templates')

    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('dag_template.tmpl')

    rendered_code = template.render(
        pipeline_id=pipeline_id,
        pipeline_description=pipeline_description,
        pipeline_user_name=pipeline_user_name,
        pipeline_user_login=pipeline_user_login,
        workflows=workflows,
        start_date = start_date
    )

    dag_file_path = os.path.join(output_dags_dir, f'dag_{pipeline_id}.py')
    with open(dag_file_path, "w") as f:
        f.write(rendered_code)

    print(f"DAG generated and saved in: {dag_file_path}")
'''
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
'''
def fetch_pipeline_from_api(pipeline_api_url, headers=None):
    response = requests.get(pipeline_api_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'OK' and data.get('data'):
            return data['data'][0]
        else:
            raise Exception("No valid pipeline found in API response.")
    else:
        raise Exception(f"Error accessing API: {response.status_code}")
    
def main(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pipeline_id', type=int, required=True, help='API ID.')
    parser.add_argument('--output_dags_dir', type=str, default='dags', help='DAGs output directory.')
    parser.add_argument('--output_workflow_dir', type=str, default='workflows', help='Workflow code output directory.')
    #parser.add_argument('--templates_dir', type=str, default='templates', help='Template airflow directory.')
    parser.add_argument('--config_path', type=str, default='conf/juicer-config.yaml', help='Configuration file.')
    parser.add_argument('--pipeline_api_url', type=str, default='https://dev.lemonade.org.br/api/v1/tahiti/pipelines/', help='Pipeline URL')
    parser.add_argument('--pipeline_api_token', type=str, required=True, help='')
    parser.add_argument('--lang', type=str, default='en', help='Minion messages language (i18n).')

    args = parser.parse_args()

    locales_path = os.path.join(os.path.dirname(__file__), 'i18n', 'locales')
    t = translation('messages', locales_path, [args.lang], fallback=True)
    t.install()

    pipeline_api_full_url = f"{args.pipeline_api_url}{args.pipeline_id}"
    
    headers = {
        'x-auth-token': args.pipeline_api_token,
    }
    
    try:
        pipeline = fetch_pipeline_from_api(pipeline_api_full_url, headers=headers)
        print(pipeline)
        pipeline_id = pipeline['id']
        pipeline_description = pipeline['description']
        pipeline_user_name = pipeline['user_name']
        pipeline_user_login = pipeline['user_login']
        pipeline_time = pipeline['created']
        start_date = datetime.fromisoformat(pipeline_time)

        print(pipeline_user_name)
        print(pipeline_user_login)
        print(start_date)

        workflows = []
        for step in pipeline.get('steps', []):
            workflow = step.get('workflow', {})
            workflow_id = workflow.get('id')
            if workflow_id:
                python_code_path = convert_lemonade_to_python(
                    workflow_id=workflow_id,
                    config_path=args.config_path,
                    lang=args.lang,
                    output_workflow_dir=args.output_workflow_dir
                )
                if python_code_path:
                    workflows.append({'id': workflow_id, 'python_file_path': python_code_path})
        
        if workflows:
            generate_dag(
                pipeline_id=pipeline_id,
                pipeline_description=pipeline_description,
                workflows=workflows,
                pipeline_user_name = pipeline_user_name,
                pipeline_user_login = pipeline_user_login,
                output_dags_dir=args.output_dags_dir,
                start_date = start_date
            )
        else:
            print("No valid workflow was found for this pipeline.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
   