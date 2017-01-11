class StateControlRedis:
    """
    Controls state of Workflows, Minions and Jobs in Lemonade.
    For minions, it is important to know if they are running or not and which
    state they keep.
    For workflows, it is important to avoid running them twice.
    Job queue is used to control which commands minions should execute.
    Finally, job output queues contains messages from minions to be sent to
    user interface.
    """
    START_QUEUE_NAME = 'queue_start'

    def __init__(self, redis_conn):
        self.redis_conn = redis_conn

    def pop_start_queue(self, block=True):
        if block:
            result = self.redis_conn.blpop(self.START_QUEUE_NAME)[1]
        else:
            result = self.redis_conn.lpop(self.START_QUEUE_NAME)
        return result

    def push_start_queue(self, data):
        self.redis_conn.rpush(self.START_QUEUE_NAME, data)

    def pop_job_queue(self, job_id, block=True):
        key = 'queue_job_{}'.format(job_id)
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result

    def push_job_queue(self, job_id, data):
        key = 'queue_job_{}'.format(job_id)
        self.redis_conn.rpush(key, data)

    def get_job_queue_size(self, job_id):
        key = 'queue_job_{}'.format(job_id)
        return self.redis_conn.llen(key)

    def get_workflow_status(self, workflow_id):
        key = 'record_workflow_{}'.format(workflow_id)
        return self.redis_conn.hget(key, 'status')

    def set_workflow_status(self, workflow_id, status):
        key = 'record_workflow_{}'.format(workflow_id)
        self.redis_conn.hset(key, 'status', status)

    def get_workflow_data(self, workflow_id):
        key = 'record_workflow_{}'.format(workflow_id)
        return self.redis_conn.hgetall(key)

    def get_minion_status(self, job_id):
        key = 'key_minion_job_{}'.format(job_id)
        return self.redis_conn.get(key)

    def set_minion_status(self, job_id, status, ex=30, nx=True):
        key = 'key_minion_job_{}'.format(job_id)
        return self.redis_conn.set(key, status, ex=ex, nx=nx)

    def pop_job_output_queue(self, job_id, block=True):
        key = 'queue_output_job_{job_id}'.format(job_id=job_id)
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result

    def push_job_output_queue(self, job_id, data):
        key = 'queue_output_job_{job_id}'.format(job_id=job_id)
        self.redis_conn.rpush(key, data)

    def get_job_output_queue_size(self, job_id):
        key = 'queue_output_job_{job_id}'.format(job_id=job_id)
        return self.redis_conn.llen(key)

    def pop_master_queue(self, block=True):
        key = 'queue_master'
        if block:
            result = self.redis_conn.blpop(key)[1]
        else:
            result = self.redis_conn.lpop(key)
        return result

    def push_master_queue(self, data):
        key = 'queue_master'
        self.redis_conn.rpush(key, data)

    def get_master_queue_size(self):
        key = 'queue_master'
        return self.redis_conn.llen(key)