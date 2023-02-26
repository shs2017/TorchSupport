"""
Module for programmatic CLI arguments from a data module

Classes:
    EnumOption

Functions:
    build_config_from_cli
"""

__all__ = ['EnumOption', 'build_config_from_cli']

from argparse import ArgumentParser, BooleanOptionalAction

from dataclasses import fields, MISSING
from enum import Enum

from functools import partial

class EnumOption(Enum):
    """To be used in a dataclass to generate a corresponding CLI choices"""
    def __str__(self):
        return str(self.name)

def str_to_enum(string, enum_type, ):
    """Returns the value based on the key name string"""
    try:
        return enum_type[string]
    except KeyError as exc:
        raise ValueError() from exc  # argparse expects a value error

def format_argument(name):
    """Simple formatter for a variable name in an argparse format"""
    return "--" + str(name)

def build_config_from_cli(config):
    """Maps cli arguments to the configuration dataclass"""
    parser = ArgumentParser()

    for field in fields(config):
        # Get the field values of the dataclass
        arg_name = format_argument(field.name)
        arg_type = field.type
        arg_default = field.default
        arg_required = True if field.default is MISSING else False

        if issubclass(arg_type, Enum):
            # In case of an enum allow for multiple options
            enum_arg_type = partial(str_to_enum, enum_type=arg_type)

            parser.add_argument(
                arg_name,
                type=enum_arg_type,
                default=arg_default,
                choices=list(arg_type),
                required=arg_required,
            )
        elif issubclass(arg_type, bool):
            parser.add_argument(
                arg_name, type=arg_type, default=arg_default, required=arg_required,
                action=BooleanOptionalAction
            )
        else:
            # Otherwise just use the normal types
            parser.add_argument(
                arg_name, type=arg_type, default=arg_default, required=arg_required
            )

    # build the dataclass from the argument parser values
    parsed_args = parser.parse_args()
    return config(**vars(parsed_args))
