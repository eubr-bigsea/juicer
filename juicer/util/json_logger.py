# coding=utf-8
import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "service": "JUICER",
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "pid": record.process,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(log_record, ensure_ascii=False)

