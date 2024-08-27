from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

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
    '36',
    default_args=default_args,
    description='Testee',
    schedule_interval=timedelta(days=1),
)

previous_task = None


task_1 = SparkSubmitOperator(
    task_id='spark_submit_813',
    application='/home/luiz/Documentos/lemonade/docker-lemonade/juicer/workflows/813.py',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_813_example',
    verbose=True,
    dag=dag,
)


previous_task = task_1
