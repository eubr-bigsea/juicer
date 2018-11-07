# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import argparse
import fcntl
import logging.config
import tempfile
import threading
import time

import gevent
import os
import redis
import redis.connection
from gevent import monkey
from gevent.subprocess import Popen, PIPE

monkey.patch_all()


logging.config.fileConfig('logging_config.ini')

log = logging.getLogger()
log.setLevel(logging.INFO)


def set_non_blocking(fd):
    """
    Set the file description of the given file descriptor to non-blocking.
    """
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    flags = flags | os.O_NONBLOCK
    fcntl.fcntl(fd, fcntl.F_SETFL, flags)


def execute_workflow(redis_conn, _id, shells):
    while True:
        log.info(_("Waiting for new commands in %s"), _id)
        _, cmd = redis_conn.brpop("list", timeout=0)
        cmds = cmd.split()
        _id = cmds[0]
        if _id in shells:
            sub, fw, fr = shells.get(_id)
            # log.info("Using existing shell")
        else:
            fw = tempfile.NamedTemporaryFile(delete=True)
            cmd = '/opt/spark-2.1.0-bin-hadoop2.6/bin/pyspark'
            cmd = 'python'
            fr = open(fw.name, "r")
            sub = Popen([cmd], stdout=fw, stdin=PIPE, stderr=fw)
            shells[_id] = (sub, fw, fr)
            set_non_blocking(fw)
            set_non_blocking(fr)

        sub.stdin.write(' '.join(cmds[1:]))
        sub.stdin.write('\n')
        sub.stdin.flush()
        gevent.sleep(.1)

        out_buffer = ""
        while True:
            gevent.sleep(1)
            out_buffer += fr.read()
            if out_buffer.endswith(">>> "):
                break


def publisher(redis_conn):
    return
    msgs = ['1 ls', '1 dir', '2 uname -a', '2 date', '1 uptime']
    for msg in msgs:
        log.info(msg)
        redis_conn.rpush("list", msg)
        gevent.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Configuration file")
    args = parser.parse_args()
    starttime = time.time()

    # Store opened shell sessions
    shells = {}

    # FIXME redis connection settings should be in config
    redis_conn = redis.StrictRedis()
    p = redis_conn.connection_pool

    publish = gevent.spawn(publisher, redis_conn)
    # FIXME: use config
    workers = 2
    log.info(_("Spawning %s greenlets connecting to Redis..."), workers)
    redis_greenlets = [gevent.spawn(execute_workflow, redis_conn, _id, shells)
                       for _id in range(workers)]
    # Wait until all greenlets have started and connected.
    gevent.sleep(1)

    log.info(_("# active `threading` threads: %s") % threading.active_count())
    log.info(_("# Redis connections created: %s") % p._created_connections)
    log.info(_("# Redis connections in use: %s") % len(p._in_use_connections))
    log.info(_("# Redis connections available: %s") % len(p._available_connections))
    log.info(_("Waiting for Redis connection greenlets to terminate..."))
    gevent.joinall(redis_greenlets)

    d = time.time() - starttime
    log.info(_("All Redis connection greenlets terminated. Duration: %.2f s.") % d)
    publish.kill()


if __name__ == '__main__':
    main()
