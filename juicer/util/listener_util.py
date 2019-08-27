# coding=utf-8


import json

JOB_START = 'job_start'
JOB_END = 'job_end'
TASK_START = 'task_start'
TASK_END = 'task_end'


def post_event_to_spark(spark_session, event_type, event_data):
    # transform dict to json
    json_event = json.dumps(event_data)
    # noinspection PyProtectedMember
    spark_session.sparkContext._jvm.lemonade.juicer.spark. \
        LemonadeSparkListener.post(event_type, json_event)
