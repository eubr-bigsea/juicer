from juicer.runner.control import StateControlRedis


class Minion:
    def __init__(self, redis_conn, job_id, config):
        self.redis_conn = redis_conn
        self.state_control = StateControlRedis(self.redis_conn)
        self.job_id = job_id
        self.config = config

    def process(self):
        raise NotImplementedError()
