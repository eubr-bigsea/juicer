from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

def convert_lemonade_to_python(workflow_id):
    python_file_path = "~/Documentos/lemonade/docker-lemonade/juicer/workflows"
    return python_file_path


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 7, 30),

}

dag = DAG(
    'lemonade_to_airflow',
    default_args=default_args,
    description='Pipeline Lemonade para Airflow',
    schedule_interval=timedelta(days=1),
)

# considerando uma sequencia de workflows de um piperline
workflow_ids = ['workflow_1', 'workflow_2', 'workflow_3']

previous_task = None

#  aqui as tasks são criadas dinamicamente para cada workflow
#  convert_lemonade_to_python carega o codigo já criado da execução de um workflow ṕelo id
for workflow_id in workflow_ids:
    python_file_path = convert_lemonade_to_python(workflow_id)
    
    task = SparkSubmitOperator(
        task_id=f'spark_submit_{workflow_id}',
        application=python_file_path,
        conn_id='spark_default',
        executor_cores=4,
        executor_memory='8g',
        driver_memory='4g',
        name=f'pyspark_{workflow_id}_example',
        verbose=True,
        dag=dag,
    )
    
    if previous_task:
        previous_task >> task
    
    previous_task = task
