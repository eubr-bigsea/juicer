# -*- coding: utf-8 -*-

from copy import deepcopy
from marshmallow import Schema, fields, post_load
from marshmallow.validate import OneOf
from models import *


def PartialSchemaFactory(schema_cls):
    schema = schema_cls(partial=True)
    for field_name, field in schema.fields.items():
        if isinstance(field, fields.Nested):
            new_field = deepcopy(field)
            new_field.schema.partial = True
            schema.fields[field_name] = new_field
    return schema


# region Protected\s*
# endregion

class OperationItemRequestSchema(Schema):
    id = fields.Integer(required=True)
    name = fields.String(required=True)


class TaskExecutionItemResponseSchema(Schema):
    """ JSON serialization schema """
    id = fields.Integer()
    date = fields.DateTime(required=False)
    status = fields.String(required=False,
                           validate=[OneOf(StatusExecution.__dict__.keys())])
    operation = fields.Function(lambda x:
                                {'id': x.operation_id,
                                 'name': x.operation_name})
    message = fields.String()
    std_out = fields.String()
    std_err = fields.String()
    exit_code = fields.Integer()

    @post_load
    def make_object(self, data):
        """ Deserializes data into an instance of TaskExecution"""
        return TaskExecution(**data)


class TaskExecutionListResponseSchema(TaskExecutionItemResponseSchema):
    pass


class UserItemResponseSchema(Schema):
    id = fields.Integer(required=True)
    login = fields.String(required=True)
    name = fields.String(required=True)


class WorkflowItemResponseSchema(Schema):
    id = fields.Integer(required=True)
    name = fields.String(required=True)
    tasks = fields.Nested(TaskExecutionItemResponseSchema, many=True)
    definition = fields.String()


class ExecutionItemResponseSchema(Schema):
    """ JSON serialization schema """
    id = fields.Integer(required=True)
    created = fields.DateTime(required=True, missing=func.now(),
                              default=func.now())
    started = fields.DateTime(required=False)
    finished = fields.DateTime(required=False)
    status = fields.String(required=True, missing=StatusExecution.WAITING,
                           default=StatusExecution.WAITING,
                           validate=[OneOf(StatusExecution.__dict__.keys())])
    workflow = fields.Function(lambda x: WorkflowItemResponseSchema().dump(dict(
        id=x.workflow_id, name=x.workflow_name, tasks=x.tasks_execution,
    )).data)

    user = fields.Function(lambda x: UserItemResponseSchema().load(dict(
        id=x.user_id, login=x.user_login, name=x.user_name)).data)


class ExecutionListResponseSchema(ExecutionItemResponseSchema):
    """ JSON serialization schema """
    pass


# ------------------------------------------------------------------------------
# Creation schemas
# ------------------------------------------------------------------------------

class OperationCreateRequestSchema(OperationItemRequestSchema):
    pass


class TaskParameterCreateRequestSchema(Schema):
    name = fields.String(required=True)
    value = fields.String(required=True)
    category = fields.String(required=True)


class OperationPortCreateRequestSchema(Schema):
    id = fields.Integer(required=True)
    direction = fields.String(required=True)


class TaskCreateRequestSchema(Schema):
    id = fields.Integer(required=False)
    log_level = fields.String(required=False)
    framework = fields.String(required=True)
    ports = fields.Nested(OperationPortCreateRequestSchema, many=True)
    operation = fields.Nested(OperationCreateRequestSchema)
    parameters = fields.Nested(TaskParameterCreateRequestSchema, many=True)

    # operation = fields.Nested(OperationCreateRequestSchema())
    # parameters
    @post_load
    def make_object(self, data):
        print '%%%%%%%%%', data
        return data


class WorkflowCreateResponseSchema(Schema):
    id = fields.Integer(required=True)
    name = fields.String(required=True)
    tasks = fields.Nested(TaskCreateRequestSchema, many=True)


class ExecutionCreateRequestSchema(Schema):
    """ JSON serialization schema """
    started = fields.DateTime(required=False)
    finished = fields.DateTime(required=False)
    workflow = fields.Nested(WorkflowCreateResponseSchema())
    # workflow_definition = fields.String(required=True)
    user = fields.Nested(UserItemResponseSchema, many=False)

    @post_load
    def make_object(self, data):
        """ Deserializes data into an instance of Execution"""

        params = {
            'workflow_id': data['workflow']['id'],
            'workflow_name': data['workflow']['name'],
            'user_id': data['user']['id'],
            'user_name': data['user']['name'],
            'user_login': data['user']['login'],
            'workflow_definition': json.dumps(data['workflow']),
        }
        print '>>>>>>>>>>>>', json.dumps(data['workflow'])
        params.update(data)
        params.pop("workflow")
        params.pop("user")
        return Execution(**params)


class ExecutionStatusRequestSchema(Schema):
    """ JSON schema for executing tasks """
    token = fields.String()

    @post_load
    def make_object(self, data):
        """ Deserializes data into an instance of Execution"""
        return Execution(**data)
