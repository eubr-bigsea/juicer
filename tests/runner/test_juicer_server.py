# -*- coding: utf-8 -*-
import collections
import json

import sys

import signal

import errno

import mock
from juicer.runner.juicer_server import JuicerServer
from juicer.runner.control import StateControlRedis
from mockredis.client import mock_strict_redis_client, MockRedis


def test_runner_read_start_queue_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow = {"app_id": app_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py',
                                           config_file_path='config.yaml')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == JuicerServer.STARTED

            assert mocked_popen.call_args_list[0][0][0] == [
                'nohup', sys.executable, 'faked_minions.py', '-a', app_id, '-c',
                'config.yaml']
            assert mocked_popen.called

            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert json.loads(state_control.pop_app_queue(app_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_app_output_queue(app_id)) == {
                'code': 0, 'message': 'Workflow will start soon'}


def test_runner_read_start_queue_workflow_not_started_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow = {"app_id": app_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # This workflow is being processed, should not start it again
            # state_control.set_workflow_status(workflow_id, JuicerServer.STARTED)
            server.active_minions[app_id] = '_'

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(app_id) is None
            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert state_control.get_app_queue_size(workflow_id) == 0

            x = state_control.pop_app_output_queue(app_id, False)
            assert json.loads(x) == {
                "message": 'Workflow {} should be started'.format(workflow_id),
                "code": 1000
                }


def test_runner_read_start_queue_minion_already_running_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    workflow = {"app_id": app_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(app_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert mocked_redis_conn.lpop('start') is None
            assert json.loads(state_control.pop_app_queue(app_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_app_output_queue(app_id)) == {
                'code': 0, 'message': 'Workflow will start soon'}


def test_runner_read_start_queue_missing_details_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    workflow_id = 1000
    # incorrect key, should raise exception
    workflow = {"xapp_id": app_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            # Publishes a message to process data
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(app_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                app_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(block=False) is None
            assert state_control.pop_app_queue(app_id, block=False) is None
            assert state_control.pop_app_output_queue(app_id,
                                                      block=False) is None

            assert mocked_redis_conn.hget(workflow_id, 'status') is None


def test_runner_master_queue_client_shutdown_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    app_id = 1
    ticket = {"app_id": app_id, 'reason': JuicerServer.HELP_UNHANDLED_EXCEPTION}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            with mock.patch('os.kill') as mocked_kill:
                server = JuicerServer(config, 'faked_minions.py')
                mocked_redis_conn = mocked_redis()
                # Publishes a message to process data
                state_control = StateControlRedis(mocked_redis_conn)

                # Configure minion
                status = {'app_id': app_id, 'pid': 9999}
                state_control.set_minion_status(app_id, json.dumps(status))
                error = OSError()
                error.errno = errno.ESRCH
                mocked_kill.side_effect = error
                # Publishes a message to master queue
                state_control.push_master_queue(json.dumps(ticket))

                # Start of testing
                server.read_minion_support_queue(mocked_redis_conn)

                assert state_control.get_minion_status(app_id) == 'STARTED'

                assert mocked_popen.called

                mocked_kill.assert_called_once_with(status['pid'],
                                                    signal.SIGKILL)


def test_runner_master_watch_minion_process_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }

    class PubSub:
        def __init__(self, m_redis):
            self.mocked_redis = m_redis
            self.channel = ""

        def psubscribe(self, channel):
            self.channel = channel

        def listen(self):
            for v in self.mocked_redis.pubsub_channels[self.channel]:
                yield v

    class CustomMockRedis(MockRedis):
        def __init__(self, strict=False, clock=None, load_lua_dependencies=True,
                     blocking_timeout=1000, blocking_sleep_interval=0.01,
                     **kwargs):
            super(CustomMockRedis, self).__init__(strict, clock,
                                                  load_lua_dependencies,
                                                  blocking_timeout,
                                                  blocking_sleep_interval,
                                                  **kwargs)
            self.pubsub = lambda: PubSub(self)
            self.pubsub_channels = collections.defaultdict(list)

        def publish(self, channel, message):
            self.pubsub_channels[channel].append(message)

        def script_kill(self):
            pass

    with mock.patch('redis.StrictRedis',
                    CustomMockRedis) as mocked_redis:
        server = JuicerServer(config, 'faked_minions.py')
        mocked_redis_conn = mocked_redis()

        mocked_redis_conn.pubsub_channels['__keyevent@*__:expired'].append(
            {'pattern': None, 'type': 'psubscribe',
             'channel': '__keyevent@*__:expired', 'data': 1L}
        )
        mocked_redis_conn.pubsub_channels['__keyevent@*__:expired'].append(
            {'pattern': '__keyevent@*__:expired', 'type': 'pmessage',
             'channel': '__keyevent@0__:expired', 'data': 'a'}
        )
        # Configure minion

        # Start of testing
        server.watch_minion_process(mocked_redis_conn)


def test_multiple_jobs_single_app():
    """ TODO
    - Start a juicer server
    - Instanciate a minion for an application
    - Submit more than one job to the same (workflow_id,app_id)
    - Assert that just one minion was launched and that these jobs shared the
      same spark_session
    """
    assert True

def test_multiple_jobs_multiple_apps():
    """ TODO
    - Start a juicer server
    - Instanciate two minions for two different aplications
    - Submit jobs for both minions
    - Assert that two minions were launched and two spark_sessions were created
    """
    assert True


