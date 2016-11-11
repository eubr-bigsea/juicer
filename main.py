
from workflow import Workflow
from IO import IO
from juicer.spark.control import Spark
from juicer.compss.control import Compss


if __name__ == '__main__':

    # Read the parameters
    io = IO()

    # Create the workflow, sort the tasks and plot the graph (optional)
    workflow = Workflow(io.args['json'])
    workflow.read_json()
    workflow.sort_tasks()
    workflow.print_workflow()
    #workflow.plot_workflow(io.args['graph_outfile'])

    if workflow.workflow['workflow']['framework'].lower() == "spark":
        spark_instance = Spark(io.args['outfile'], workflow.workflow,
        workflow.sorted_tasks)
        spark_instance.execution()


    elif workflow.workflow['workflow']['framework'].lower() == "compss":
        compss_instance = Compss(io.args['outfile'], workflow.workflow,
        workflow.sorted_tasks)
        compss_instance.execution()