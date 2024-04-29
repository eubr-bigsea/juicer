class ChartBuilder:
    def __init__(self):
        self.chart = {
            "parameters": {},
            "named_inputs": {},
            "named_outputs": {}
        }

    def set_type(self, chart_type):
        self.chart["parameters"]["type"] = chart_type
        return self

    def hide_legend(self, display_legend):
        self.chart["parameters"]["display_legend"] = display_legend
        return self

    def set_task_id(self, task_id):
        self.chart["parameters"]["task_id"] = task_id
        return self

    def add_named_input(self, name, input_data):
        self.chart["named_inputs"][name] = input_data
        return self

    def add_named_output(self, name, output_data):
        self.chart["named_outputs"][name] = output_data
        return self

    def set_x(self, x_builder):
        self.x = x_builder

    def build(self):
        return self.chart


class XBuilder:
    def __init__(self):
        self.x_parameter = {}

    def binning(self, value):
        self.binning = value
        return self
    def bins(self, value):
        self.bins = value
        return self
    def binSize(self, value):
        self.binSize = value
        return self
    def emptyBins(self, value):
        self.emptyBins = value
        return self
    def multiplier(self, value):
        self.multiplier = value
        return self
    def decimal_places(self, value):
        self.decimal_places = value
        return self
    def prefix(self, value):
        self.prefix = value
        return self
    def suffix(self, value):
        self.suffix = value
        return self
    def label(self, value):
        self.label = value
        return self
    def max_displayed(self, value):
        self.max_displayed = value
        return self
    def group_others(self, value):
        self.group_others = value
        return self
    def sorting(self, value):
        self.sorting = value
        return self
    def attribute(self, value):
        self.attribute = value
        return self


class YParameterBuilder:
    def __init__(self):
        self.y_parameter = {}

    def add_attribute(self, attribute):
        self.y_parameter["attribute"] = attribute
        return self

    def add_aggregation(self, aggregation):
        self.y_parameter["aggregation"] = aggregation
        return self

    # Add other properties...

    def add_to_chart(self):
        if "y" not in self.chart_builder.chart["parameters"]:
            self.chart_builder.chart["parameters"]["y"] = []
        self.chart_builder.chart["parameters"]["y"].append(self.y_parameter)
        return self.chart_builder


class XAxisBuilder:
    def __init__(self):
        self.x_axis = {}

    def add_lower_bound(self, lower_bound):
        self.x_axis["lowerBound"] = lower_bound
        return self

    # Add other properties...

    def add_to_chart(self):
        self.chart_builder.chart["parameters"]["x_axis"] = self.x_axis
        return self.chart_builder


class YAxisBuilder:
    def __init__(self):
        self.y_axis = {}

    def add_lower_bound(self, lower_bound):
        self.y_axis["lowerBound"] = lower_bound
        return self

    # Add other properties...

    def add_to_chart(self):
        self.chart_builder.chart["parameters"]["y_axis"] = self.y_axis
        return self.chart_builder

