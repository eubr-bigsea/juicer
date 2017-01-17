# coding=utf-8
from __future__ import print_function
import json
from textwrap import dedent

import mock
import os
from juicer.runner.control import StateControlRedis
from juicer.spark.spark_minion import SparkMinion
from mockredis import mock_strict_redis_client

config = {
    'juicer': {
        'servers': {
            'redis_url': 'redis://invalid:2923'
        }
    }
}


# noinspection PyProtectedMember
def test_minion_ping_success():
    job_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                             config=config)
        minion._perform_ping()

        state_control = StateControlRedis(redis_conn)

        assert json.loads(state_control.get_minion_status(job_id)) == {
            'status': 'READY', 'pid': os.getpid()}
        assert state_control.get_job_output_queue_size(job_id) == 0


# noinspection PyProtectedMember
def test_minion_generate_output_success():
    job_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                             config=config)

        state_control = StateControlRedis(redis_conn)

        msgs = ["Message being sent \n{}", {'msg': 'Dictionary being sent'}]
        for msg in msgs:
            minion._generate_output(msg)
            result = json.loads(
                state_control.pop_job_output_queue(job_id, False))
            assert result.keys() == ['date', 'message', 'job_id']
            assert result['job_id'] == job_id
            assert result['message'] == msg
        assert state_control.get_job_output_queue_size(job_id) == 0


# noinspection PyProtectedMember
def test_minion_perform_execute_success():
    job_id = 897447
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    # noinspection PyUnusedLocal
    def side_effect(w, g, c, out):
        print(dedent("""
            def main():
                return {
                    "xyz-647": ("df", 27.27)
                }"""), file=out)

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = side_effect

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                                 config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            state_control.push_job_queue(job_id, json.dumps({'workflow': data}))

            minion._perform_execute()

            assert minion.state == {"xyz-647": ("df", 27.27)}, 'Invalid state'

            assert state_control.get_job_output_queue_size(
                job_id) == 1, 'Wrong number of output messages'
