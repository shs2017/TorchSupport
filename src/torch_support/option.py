"""
Module for programmatic model creation

Classes:
    RegularOption
    SyncOption
    ProductOption
    RangeOption
    CopyOption
    NamedOption
    OptionBuilder
"""

__all__ = ['RegularOption', 'SyncOption', 'ProductOption',
           'RangeOption', 'CopyOption', 'NamedOption', 'OptionBuilder']

from typing import Sequence
from itertools import chain, cycle, product

from copy import deepcopy

class BaseOption:
    def __init__(self):
        pass

class RegularOption(BaseOption):
    """Wrapper for values that aren't instances of
       an option subclass"""
    def __init__(self, val: Sequence):
        super().__init__()
        self.val = cycle([val])

    @classmethod
    def apply(cls, *option_list):
        return zip(*map(lambda x : x.val, option_list))

class SyncOption(BaseOption):
    """When the builder iterates through the model 
       these options just iterate through one by
       one in order regardless of the other 
       options"""
    def __init__(self, val: Sequence):
        super().__init__()
        self.val = iter(val)

    @classmethod
    def apply(cls, *option_list):
        return zip(*map(lambda x : x.val, option_list))

class ProductOption(BaseOption):
    """When the builder iterates through the model
       these options iterate through product-wise.
       That is, each iteration will go through one
       element in the set of the cartesian product
       of the ProductionOption elements"""
    def __init__(self, val: Sequence):
        super().__init__()
        self.val = iter(val)

    @classmethod
    def apply(cls, *option_list):
        return product(*map(lambda x : x.val, option_list))

class RangeOption(BaseOption):
    """When the builder iterates through the model
       these options iterate through product-wise.
       That is, each iteration will go through one
       element in the set of the cartesian product
       of the ProductionOption elements"""
    def __init__(self, a, b=None, step=1):
        super().__init__()
        start = a
        end = b

        if not b:
            self.val = range(a)
        else:
            self.val = range(a, b, step)

    @classmethod
    def apply(cls, *option_list):
        return product(*map(lambda x : x.val, option_list))

class CopyOption(BaseOption):
    """When the builder iterates through the model
       these options iterate through product-wise.
       That is, each iteration will go through one
       element in the set of the cartesian product
       of the ProductionOption elements"""
    def __init__(self, val: Sequence):
        super().__init__()
        self.val = map(lambda x: deepcopy(x), cycle([val]))

    @classmethod
    def apply(cls, *option_list):
        return zip(*map(lambda x : x.val, option_list))


class NamedOption(BaseOption):
    """Takes an option and combines its
       name in the iter"""
    def __init__(self, name, option):
        super().__init__()

        self.val = map(
            lambda val : (name, val),
            option.val
        )

        self.option = option

    @classmethod
    def apply(cls, *option_list):
        f = option_list[0].option.apply
        return f(*option_list)

    def __str__(self):
        return f'(name= {name}, {val=})'


class OptionBuilder:
    def __init__(self, options):
        self.options = self._build(options)

    def _group(self, option):
        grouped_options = {}

        for name, value in option.items():
            if not isinstance(value, BaseOption):
                value = RegularOption(value)

            key = str(value.__class__.__name__)

            if key not in grouped_options:
                grouped_options[key] = []

            named_option = NamedOption(name, value)
            grouped_options[key].append(named_option)

        return grouped_options

    def _parse(self, options):
        res = []
        for _, option in options.items():
            res.append(NamedOption.apply(*option))
        return zip(*res)

    def _build(self, option):
        return self._parse(self._group(option))

    def __iter__(self):
        return self

    def __next__(self):
        return dict(chain(*next(self.options)))
