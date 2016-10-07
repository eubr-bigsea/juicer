# -*- coding: utf-8 -*-
import json

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, \
    Enum, DateTime, Numeric, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, backref

db = SQLAlchemy()


# noinspection PyClassHasNoInit
class StatusExecution:
    COMPLETED = 'COMPLETED'
    WAITING = 'WAITING'
    INTERRUPTED = 'INTERRUPTED'
    CANCELED = 'CANCELED'
    RUNNING = 'RUNNING'
    ERROR = 'ERROR'
    PENDING = 'PENDING'


class Execution(db.Model):
    """ Records a workflow execution """
    __tablename__ = 'execution'

    # Fields
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False, default=func.now())
    started = Column(DateTime)
    finished = Column(DateTime)
    status = Column(Enum(*StatusExecution.__dict__.keys(),
                         name='StatusExecutionEnumType'), nullable=False,
                    default=StatusExecution.WAITING)
    workflow_id = Column(Integer, nullable=False)
    workflow_name = Column(String(200), nullable=False)
    workflow_definition = Column(Text, nullable=False)
    user_id = Column(Integer, nullable=False)
    user_login = Column(String(50), nullable=False)
    user_name = Column(String(200), nullable=False)
    # Associations
    tasks_execution = relationship("TaskExecution", back_populates="execution")

    def __unicode__(self):
        return self.created

    def __repr__(self):
        return '<Instance {}: {}>'.format(self.__class__, self.id)


class TaskExecution(db.Model):
    """ Records a task execution """
    __tablename__ = 'task_execution'

    # Fields
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    status = Column(Enum(*StatusExecution.__dict__.keys(),
                         name='StatusExecutionEnumType'), nullable=False)
    task_id = Column(Integer, nullable=False)
    operation_id = Column(Integer, nullable=False)
    operation_name = Column(String(200), nullable=False)
    # Associations

    execution_id = Column(Integer,
                          ForeignKey("execution.id"), nullable=False)
    execution = relationship("Execution", foreign_keys=[execution_id])

    def __unicode__(self):
        return self.date

    def __repr__(self):
        return '<Instance {}: {}>'.format(self.__class__, self.id)
