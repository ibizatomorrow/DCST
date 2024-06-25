import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from STGNN.pemsd7.DCRNN.util import read_json, write_json
from pathlib import Path


class ConfigParser:
    def __init__(self):
        # parse default and custom cli options

        self.resume = Path('../STGNN/pemsd7/DCRNN/metrla.pth')
        self.cfg_fname = self.resume.parent / 'config.json'

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = config


    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
