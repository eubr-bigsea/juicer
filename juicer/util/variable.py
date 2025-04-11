import datetime
import re
import typing
from gettext import gettext

VARIABLE_RE = re.compile(r"\$\{([_A-Za-z][_A-Za-z0-9]*)(?:\|(.+?))?\}")
VARIABLE_NOT_REPLACED_RE = re.compile(r"\$\{.+?\}")
# Supported formats
date_formats = [
    "%Y-%m-%d",  # Date only
    "%H:%M:%S",  # Time only
    "%Y-%m-%d %H:%M:%S",  # Date and time
    "%d/%m/%Y",  # Date with slashes (day first)
    "%d/%m/%Y %H:%M:%S",  # Date and time with slashes (day first)
    "%Y/%m/%d",  # Date with slashes (year first)
    "%Y/%m/%d %H:%M:%S",  # Date and time with slashes (year first)
]


def _parse_date(date_str: str) -> datetime.datetime:
    """Try to parse date using supported formats"""
    for date_format in date_formats:
        try:
            return datetime.datetime.strptime(date_str, date_format)
        except Exception:
            continue
    return None


def _replace(value: str, all_vars: typing.Dict, var_re: re.Pattern):
    def replacer(match):
        variable = match.group(1)  # Extracts content between ${}
        param = match.group(2)
        if param:
            # Variables that support parameters are identified by
            # $name and associated to a lambda function
            value = all_vars.get(variable)
            return all_vars.get(f"${variable}")(value, param)
        else:
            return str(all_vars.get(variable))

    return var_re.sub(replacer, value)


def handle_variables(
    user: any,
    values: typing.List[str],
    custom_vars: typing.Dict = None,
    parse_date=True
) -> typing.List[str]:
    """
    Handles variable substitution
    FIXME: Refactoring code in workflow.py module to use this function
    """
    result = []
    now = datetime.datetime.now()

    all_vars = {
        "date": now.strftime("%Y-%m-%d"),
        # Special variables, used in conjunction with a formatting pattern.
        # Variables that support parameters are identified by $name and
        # are associated to a lambda function
        "$date": lambda value, fmt: datetime.datetime.now().strftime(fmt),
        # $ref is a special variable, used by pipelines
        "$ref": lambda value, fmt: value.strftime(fmt),
    }
    if user:
        all_vars.update(
            {
                "user_login": user.login,
                "user_name": user.name,
                "user_email": user.login,  # FIXME
                "user_id": str(user.id),
            }
        )

    if custom_vars:
        for name, value in custom_vars.items():
            if parse_date:
                date_value = _parse_date(value)
                if date_value:
                    value = date_value
            all_vars[name] = value
    for value in values:
        if value is None:
            result.append(None)
        else:
            new_value = _replace(value, all_vars, VARIABLE_RE)
            result.append(new_value)
            missing = VARIABLE_NOT_REPLACED_RE.findall(new_value)
            if missing:
                raise ValueError(
                    gettext(
                        "Not all variables were expanded. "
                        "Please, check informed values: {}".format(
                            ", ".join(missing)
                        )
                    )
                )
    return result
