from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 7, 30),
    'retries': 1,
}

dag = DAG(
    'spark_submit_code01_dag',
    default_args=default_args,
    description='Executar o script code01.py usando SparkSubmitOperator',
    schedule_interval='@daily',
)

spark_submit_task = SparkSubmitOperator(
    task_id='spark_submit_code01_task',
    application='/Documentos/lemonade/docker-lemonade/juicer/workflows"/code01.py', 
    conn_id='spark_default',  
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_code01_example',
    verbose=True,
    dag=dag,
)

spark_submit_task
