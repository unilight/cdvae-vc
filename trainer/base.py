import tensorflow as tf
import numpy as np
import logging, os

class Trainer(object):
    def __init__(self, loss, valid, arch, args, dirs):
        self.loss = loss
        self.valid = valid
        self.arch = arch
        self.args = args
        self.dirs = dirs
        self.opt = self._optimize()

    def _optimize(self):
        """ To be implemented by child class
            Should rovide the following operators:
            opt: update operator
            global_step: global step
        """
        return {}

