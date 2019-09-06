# coding=utf-8

# Comm protocol between clients, JuicerServer and Minion ##

# EXECUTE messages request the execution of a new job
# {
#   'workflow_id': <workflow identifier>
#   'app_id': <app identifier, representing an workflow instance>
#   'job_id': <job identifier, representing a command submited to the app>
#   'type': 'execute'
#   'app_configs': <key value pairs as app resource and env configurations>
#   'workflow': <workflow that will be transpiled and executed>
# }
EXECUTE = 'execute'

# SCRIPT request the execution of a special script,
# in general something related to researching branches
# {
#        'execution_id': <execution id>,
#        'script': <script name>,
#        'params: <object with params, as a dict in Python>
# }
# 
SCRIPT = 'script'

# TERMINATE messages request the termination of a minion
# NOTE: if 'job_id' is present then we will stop all current jobs and keep the
# app alive, i.e., its resource context
# {
#   'workflow_id': <workflow identifier>
#   'app_id': <app identifier, representing an workflow instance>
#   ['job_id': <job identifier, representing a command submited to the app>]
#   'type': 'terminate'
# }
TERMINATE = 'terminate'
