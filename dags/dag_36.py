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
    task_id='spark_submit_702',
    application='/home/luiz/Documentos/lemonade/docker-lemonade/juicer/workflows/workflow_702.py',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_702_example',
    verbose=True,
    dag=dag,
)


previous_task = task_1

task_2 = SparkSubmitOperator(
    task_id='spark_submit_152',
    application='None',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_152_example',
    verbose=True,
    dag=dag,
)


previous_task >> task_2

previous_task = task_2

task_3 = SparkSubmitOperator(
    task_id='spark_submit_692',
    application='None',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_692_example',
    verbose=True,
    dag=dag,
)


previous_task >> task_3

previous_task = task_3

task_4 = SparkSubmitOperator(
    task_id='spark_submit_694',
    application='None',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_694_example',
    verbose=True,
    dag=dag,
)


previous_task >> task_4

previous_task = task_4

task_5 = SparkSubmitOperator(
    task_id='spark_submit_692',
    application='None',
    conn_id='spark_default',
    executor_cores=4,
    executor_memory='8g',
    driver_memory='4g',
    name='pyspark_692_example',
    verbose=True,
    dag=dag,
)


previous_task >> task_5

previous_task = task_5
