# coding=utf-8

import uuid


class Deployment(object):
    def __init__(self):
        self.tasks = []
        self.flows = []

    def add_task(self, task):
        self.tasks.append(task)

    def add_flow(self, flow):
        self.flows.append(flow)


class DeploymentFlow(object):
    def __init__(self, source_id, source_port, source_port_name,
                 target_id, target_port, target_port_name):
        self.source_id = source_id
        self.source_port = source_port
        self.source_port_name = source_port_name
        self.target_id = target_id
        self.target_port = target_port
        self.target_port_name = target_port_name


class DeploymentFormField(object):
    def __init__(self, name, category, value):
        self.value = value
        self.category = category
        self.name = name


class DeploymentOperation(object):
    def __init__(self, op_id, slug):
        self.id = op_id
        self.slug = slug


class DeploymentTask(object):
    def __init__(self, original_id):
        self.operation = None
        self.forms = {}
        self.original_id = original_id
        self.id = str(uuid.uuid1())
        self.top = 0
        self.left = 0
        self.z_index = 0

    def set_operation(self, op_id=None, slug=None):
        self.operation = DeploymentOperation(op_id, slug)
        return self

    def add_field(self, name, category, value):
        self.forms[name] = DeploymentFormField(name, category, value)
        return self

    def set_properties(self, forms):
        for name, category, value in forms:
            self.add_field(name, category, value)
        return self

    def set_pos(self, top, left, z_index):
        self.top = top
        self.left = left
        self.z_index = z_index
        return self
