from textwrap import dedent

class Kmeans:

    def __init__(self, parameters):
        print "\n\n",parameters,"\n\n"
        self.k = parameters['k']
        self.num_frag = parameters['num_frag']
        self.max_iterations = parameters['max_iterations']
        self.epsilon = parameters['epsilon']
        self.init_mode = parameters['init_mode']

    def generate_code(self):
        code = """
            import juicer.comps.operation.kmeans
            kmeans({}, {}, {}, {}, {}, "{}")
            """.format("data", self.k, self.num_frag,
                       self.max_iterations, self.epsilon,
                       self.init_mode)
        return dedent(code)



class DataReader:
    def __init__(self, parameters):
        self.generate_code()

    def generate_code(self):
        return ("DATA READER")
