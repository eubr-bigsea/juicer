# -*- coding: utf-8 -*-}
import datetime

from flask import request, current_app
from flask_restful import Resource

from app_auth import requires_auth
from models import db, Execution
from schema import *


class ExecutionListApi(Resource):
    """ REST API for listing class Execution """

    @staticmethod
    @requires_auth
    def get():
        only = ('id', 'name') \
            if request.args.get('simple', 'false') == 'true' else None
        executions = Execution.query.all()
        return ExecutionListResponseSchema(many=True, only=only).dump(
            executions).data

    @staticmethod
    @requires_auth
    def post():
        result, result_code = dict(
            status="ERROR", message="Missing json in the request body"), 401
        if request.json is not None:
            request_schema = ExecutionCreateRequestSchema()
            response_schema = ExecutionItemResponseSchema()
            form = request_schema.load(request.json)
            if form.errors:
                result, result_code = dict(
                    status="ERROR", message="Validation error",
                    errors=form.errors, ), 401
            else:
                try:
                    execution = form.data
                    # fill task execution records
                    print request.json
                    for t in request.json.get('workflow', {}).get('tasks', []):
                        execution.tasks_execution.append(TaskExecution(
                            date=datetime.datetime.now(),
                            status=StatusExecution.PENDING,
                            task_id=t['id'],
                            operation_id=t['operation']['id'],
                            operation_name=t['operation']['name'],
                        ))
                    db.session.add(execution)
                    db.session.commit()
                    result, result_code = response_schema.dump(
                        execution).data, 200
                except Exception, e:
                    result, result_code = dict(status="ERROR",
                                               message="Internal error"), 500
                    if current_app.debug:
                        result['debug_detail'] = e.message
                    db.session.rollback()

        return result, result_code


class ExecutionDetailApi(Resource):
    """ REST API for a single instance of class Execution """

    @staticmethod
    @requires_auth
    def get(execution_id):
        execution = Execution.query.get(execution_id)
        if execution is not None:
            return ExecutionItemResponseSchema().dump(execution).data
        else:
            return dict(status="ERROR", message="Not found"), 404
