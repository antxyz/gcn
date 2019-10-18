from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf


class AGGCN(tf.keras.Model):

    def __init__(self,opt):
        super(ResNet, self).__init__()
        self.block_MLP = opt['nums_block']
        self.layers = []
        for i in range(self.block_MLP):
            self.layers.append(MLPBlock())

    def call(self, inputs):

        for i in range(self.block_MLP):
            self.layers[i](inputs)

        return output 
