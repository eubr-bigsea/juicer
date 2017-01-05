# coding=utf-8
"""
"
"""
import multiprocessing
import argparse
import importlib
import os
import time
import pyspark

import sys


class JuicerServer:
    """
    Server
    """

    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.consumer_process = None
        self.executor_process = None

    def consume(self):
        filename = '/mnt/spark/teste.py'
        cached_stamp = os.stat(filename).st_mtime
        p = multiprocessing.current_process()
        print 'Starting:', p.name, p.pid
        sys.stdout.flush()
        while True:
            stamp = os.stat(filename).st_mtime
            if stamp != cached_stamp:
                cached_stamp = stamp
                time.sleep(2)
                self.queue.put(filename)
        time.sleep(2)
        print 'Exiting :', p.name, p.pid
        sys.stdout.flush()

    def execute(self):
        p = multiprocessing.current_process()
        print 'Starting:', p.name, p.pid
        sys.stdout.flush()
        modules = {}
        while True:
            filepath = self.queue.get()
            path = os.path.dirname(filepath)
            name = os.path.basename(filepath)

            m = modules.get(filepath)
            if m is None:
                if path not in sys.path:
                    sys.path.append(path)
                print 'loading'
                m = importlib.import_module(name.split('.')[0])
                modules[filepath] = m
            else:
                print 'Reloading'
                reload(m)

            print 'Starting processing'
            m.main()
            print 'Finished processing'

        print 'Exiting :', p.name, p.pid
        sys.stdout.flush()

    def start(self):
        self.consumer_process = multiprocessing.Process(name="consumer",
                                                        target=self.consume)
        self.executor_process = multiprocessing.Process(name="executor",
                                                        target=self.execute)
        self.consumer_process.daemon = False
        self.executor_process.daemon = False
        self.consumer_process.start()
        self.executor_process.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="Listen port")
    args = parser.parse_args()

    print 'OK, iniciando'
    server = JuicerServer()
    server.start()
