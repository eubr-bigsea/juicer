import argparse


class IO:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Lemonade module that receive \
            the workflow Json and generate the Spark code.')
        parser.add_argument('-j', '--json', help='Json file describing the Lemonade \
            workflow', required=False)
        parser.add_argument('-o', '--outfile', help='Outfile name to receive the Spark \
            code', required=False)
        parser.add_argument('-g', '--graph_outfile', help='Outfile name to plot the  \
            workflow graph', required=False)
        parser.add_argument("-w", "--workflow", type=int, required=True,
                            help="Workflow identification number")
        self.args = vars(parser.parse_args())
