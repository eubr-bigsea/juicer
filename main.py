import json

import requests
from IO import IO
from juicer.spark.transpiler import SparkTranspiler
from juicer.compss.control import Compss
from juicer.workflow.workflow import Workflow



if __name__ == '__main__':

    # Read the parameters
    io = IO()

    # Create the workflow, sort the tasks and plot the graph (optional)
    r = requests.get(
        'http://beta.ctweb.inweb.org.br/tahiti/workflows/{}?token=123456'.format(
            io.args['workflow']))
    workflow = Workflow(None)
    workflow_def = json.loads(r.text)
    workflow.set_workflow(workflow_def)
    workflow.sort_tasks()

    if not io.args['graph_outfile'] is None:
        workflow.plot_workflow(io.args['graph_outfile'])

    if workflow.workflow.get('framework', 'spark').lower() == "spark":
        spark_instance = SparkTranspiler(io.args['outfile'], workflow.workflow,
                                         workflow.sorted_tasks)
        spark_instance.transpile()

    elif workflow.workflow.get('framework').lower() == "compss":
        compss_instance = Compss(io.args['outfile'], workflow.workflow,
                                 workflow.sorted_tasks)
        compss_instance.execution()
