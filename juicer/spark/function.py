from wrap import dedent

class Function():
    def set_io(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class Replace(Function):
    def __init__(self, parameters, inputs, outputs):
        self.target_column = parameters['target_column']
        self.old_content = parameters['old_content']
        self.new_content = parameters['new_content']
        #self. = parameters['replacement']
        #try:
        #    self.new_columns = parameters['new_col']
        #except KeyError:
        #    self.new_columns = self.target_column
        self.set_io(inputs, outputs)

    def generate_code(self):
        code = '''
            {} = {}.withColumn({}, regexp_replace(col({}), "{}", "{}"))
        '''.format(output[0], input[0], self.target_column, self.target_column,
                self.old_content, self.new_content)
        return dedent(code)