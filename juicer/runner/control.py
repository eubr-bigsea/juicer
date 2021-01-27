# coding=utf-8
class StateControlRedis:
    """
    Controls state of Workflows, Minions and Jobs in Lemonade.
    For minions, it is important to know if they are running or not and which
    state they keep.
    For workflows, it is important to avoid running them twice.
    Job queue is used to control which commands minions should execute.
    Finally, app output queues contains messages from minions to be sent to
    user interface.
    """
    START_QUEUE_NAME = 'queue_start'
    SCRIPT_QUEUE_NAME = 'queue_script'

    QUEUE_APP = 'queue_app_{}'

    def __init__(self, redis_conn):
        self.redis_conn = redis_conn

    def pop_queue(self, queue, block=True, timeout=0):
        if block:
            result = self.redis_conn.blpop(queue, timeout=timeout)
            if result is not None:
                result = result[1]
        else:
            result = self.redis_conn.lpop(queue)
        if result is not None:
            if isinstance(result, bytes):
                return result.decode("utf-8")
            else:
                return result
        else:
            return None

    def push_queue(self, queue, data, ttl=0):
        self.redis_conn.rpush(queue, data)
        if ttl > 0:
            self.redis_conn.expire(queue, ttl)

    def pop_start_queue(self, block=True):
        return self.pop_queue(self.START_QUEUE_NAME, block)

    def push_start_queue(self, data):
        self.push_queue(self.START_QUEUE_NAME, data)

    def pop_app_queue(self, app_id, block=True, timeout=0):
        return self.pop_queue(self.QUEUE_APP.format(app_id), block, timeout)

    def push_app_queue(self, app_id, data):
        self.push_queue(self.QUEUE_APP.format(app_id), data)

    def get_app_queue_size(self, app_id):
        key = self.QUEUE_APP.format(app_id)
        return self.redis_conn.llen(key)

    def get_workflow_status(self, workflow_id):
        key = 'record_workflow_{}'.format(workflow_id)
        result = self.redis_conn.hget(key, 'status')
        return result if result else None

    def set_workflow_status(self, workflow_id, status):
        key = 'record_workflow_{}'.format(workflow_id)
        self.redis_conn.hset(key, 'status', status)

    def get_workflow_data(self, workflow_id):
        key = 'record_workflow_{}'.format(workflow_id)
        return self.redis_conn.hgetall(key)

    def get_minion_status(self, app_id):
        key = 'key_minion_app_{}'.format(app_id)
        result = self.redis_conn.get(key)
        return result if result else None

    def set_minion_status(self, app_id, status, ex=30, nx=True):
        key = 'key_minion_app_{}'.format(app_id)
        return self.redis_conn.set(key, value=status, ex=ex, nx=nx)

    def unset_minion_status(self, app_id):
        key = 'key_minion_app_{}'.format(app_id)
        return self.redis_conn.delete(key)

    def pop_app_output_queue(self, app_id, block=True):
        key = 'queue_output_app_{app_id}'.format(app_id=app_id)
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result if result else None

    def push_app_output_queue(self, app_id, data):
        key = 'queue_output_app_{app_id}'.format(app_id=app_id)
        self.redis_conn.rpush(key, data)

    def get_app_output_queue_size(self, app_id):

        key = 'queue_output_app_{app_id}'.format(app_id=app_id)
        return self.redis_conn.llen(key)

    def pop_master_queue(self, block=True):
        key = 'queue_master'
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result if result else None

    def push_master_queue(self, data):
        key = 'queue_master'
        self.redis_conn.rpush(key, data)

    def get_master_queue_size(self):
        key = 'queue_master'
        return self.redis_conn.llen(key)

    # def pop_app_delivery_queue(self, app_id, block=True):
    #     key = 'queue_delivery_app_{app_id}'.format(app_id=app_id)
    #     if block:
    #         result = self.redis_conn.blpop(key)[1]
    #     else:
    #         result = self.redis_conn.lpop(key)
    #     return result

    # def push_app_delivery_queue(self, app_id, data):
    #     key = 'queue_delivery_app_{app_id}'.format(app_id=app_id)
    #     self.redis_conn.rpush(key, data)

    # def get_app_delivery_queue_size(self, app_id):
    #     key = 'queue_delivery_app_{app_id}'.format(app_id=app_id)
    #     return self.redis_conn.llen(key)
    def pop_script_queue(self, block=True):
        return self.pop_queue(self.SCRIPT_QUEUE_NAME, block)

    def shutdown(self):
        self.redis_conn.close()

