from juicer.compss import operation


class Compss:
    def __init__(self, outfile, workflow, tasks):
        self.output = open(outfile, "w")
        self.workflow = workflow
        self.tasks = tasks
        self.classes = {}
        self.assign_operations()

    def execution(self):
        for task in self.tasks:
            parameters = {}
            for parameter in task['parameters']:
                if parameter['category'] == "EXECUTION":
                    parameters[parameter['name']] = parameter['value']

            class_name = self.classes[task['operation']['name']]
            instance = class_name(parameters)
            self.output.write(instance.generate_code())

    def assign_operations(self):
        self.classes['KMEANS'] = operation.Kmeans
        self.classes['DATA_READER'] = operation.DataReader
