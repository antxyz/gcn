from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf

#AGGCN block
class MLPBlock(layers.Layer):
    def __init__(self,):
        super().__init__()
        self.heads=opt['heads']
        # 密集连接层的层数
        self.layers_dense=opt['layers_dense']
        self.linear_1 = MutiHeadsAttn()
        self.linear_2 = DenseConLayer()
        self.linear_3 = Linear()

    def call(self,inputs):
        
        output = self.linear_1(inputs)
        output = self.linear_2(output)
        output = self.linear_3(output)

        return output

# 线性组合
class LinearCom(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self,input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)

    def call(self,input):

        return output

#多头注意力层
class MutiHeadsAttn(layers.Layer):
    def __init__(self):
        super().__init__(self)

    def build(self,input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)
                                
    def call(self,inputs):

        return tf.matmul(inputs, self.w) + self.b

class DenseConBlock(layers.Layer):
    def __init__(self,opt):
        super().__init__()
        # 密集连接层的块数
        self.block_dense=opt['heads']
        #保存密集连接块
        self.block = []
        for i in range(self.block_dense):
            self.block.append(DenseConLayer())

    def call(self,inputs,adj_list):
        outputs = []
        for i in range(len(adj_list)):
            output = self.block[i](inputs,adj_list[i])
            outputs.append(output)
        # 返回（self.block_dense个）密集连接层的隐藏表示列表
        return outputs

# 密集连接层  
# 接收：
#       子层数，临界矩阵和输入词向量
class DenseConLayer(layers.Layer):
    def __init__(self,opt):
        super().__init__()
        # 密集连接层的子层数
        self.layers_subdense=opt['heads']
        self.layers=[]
        for i in range(self.layers_subdense):
            self.layers.append(GraphConvLayer(opt))
        
    # def build(self,input_shape):
    #     self.w = self.add_weight(shape=(input_shape[-1], self.units),
    #                             initializer='random_normal',
    #                             trainable=True)
    #     self.b = self.add_weight(shape=(self.units,),
    #                             initializer='random_normal',
    #                             trainable=True)   

    def call(self,inputs,adj):
        outputs = [inputs]
        for i in range(len(self.layers)):
            output_layer = self.layers[i](inputs,adj)
            outputs.append(output_layer)
            inputs = tf.concat(outputs,0)
            
        return outputs


class GraphConvLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True)

    def call(self,inputs,adj):
        # 创建单位矩阵
        matrix_I=tf.eye(adj.shape[0])
        # 邻接矩阵加上自循环
        adj+=matrix_I
        # 获取邻接矩阵行累加向量
        sum_row=tf.expand_dims(tf.reduce_sum(adj,1),1)
        # 将邻接矩阵做归一化处理
        adj/=sum_row
        # 对输入进行计算
        W_X = tf.matmul(self.w,inputs)
        A_W_X=tf.matmul(W_X,tf.transpose(adj))
        output=tf.nn.relu.A_W_X+self.b
        return output


