import argparse
import json

import requests
from IO import IO
from juicer.spark.transpiler import SparkTranspiler
from juicer.compss.control import Compss
from juicer.workflow.workflow import Workflow

if __name__ == '__main__':

    # Read the parameters
    parser = argparse.ArgumentParser(description='Lemonade module that receive \
        the workflow Json and generate the Spark code.')
    parser.add_argument("-w", "--workflow", type=int, required=True,
                        help="Workflow identification number")

    parser.add_argument("-s", "--service", type=bool, required=False,
                        action="store_true",
                        help="Indicates if workflow will run as a service")
    args = parser.parse_args()

    # Create the workflow, sort the tasks and plot the graph (optional)
    r = requests.get(
        ('http://beta.ctweb.inweb.org.br'
         '/tahiti/workflows/{}?token=123456').format(args['workflow']))
    workflow = Workflow(json.loads(r.text))

    #if not args['graph_outfile'] is None:
    #    workflow.plot_workflow(args['graph_outfile'])

    if workflow.workflow.get('framework', 'spark').lower() == "spark":
        spark_instance = SparkTranspiler(args['outfile'], workflow.workflow)
        spark_instance.transpile()

    elif workflow.workflow.get('framework').lower() == "compss":
        compss_instance = Compss(args['outfile'], workflow.workflow)
        compss_instance.execution()
