from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf

class GraphConv(layers.Layer):
    def __init__(self):
        super().__init__()