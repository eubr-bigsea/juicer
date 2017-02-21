# -*- coding: utf-8 *-
__CONFIG__ = None

def set_config(config):
    global __CONFIG__
    __CONFIG__ = config

def get_config():
    global __CONFIG__
    return __CONFIG__
