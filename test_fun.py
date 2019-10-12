

import torch 


#multiply:相应位置相乘
#matmul:正常的矩阵相乘
# v1 = tf.constant(1,shape=[2,2])
# v2 = tf.constant(2,shape=[2,2])
# print([1,2][3,4])


# class GraphConvLayer(layers.Layer):
#     def __init__(self):
#         super().__init__()

#     def build(self, input_shape):
#         self.w = self.add_weight(shape=(input_shape[-1], self.units),
#                                 initializer='random_normal',
#                                 trainable=True)
#         self.b = self.add_weight(shape=(self.units,),
#                                 initializer='random_normal',
#                                 trainable=True)

#     def call(self,adj,inputs):
#         # 创建单位矩阵
#         matrix_I=tf.eye(adj.shape[0],dtype='double')
#         # 邻接矩阵加上自循环
#         adj+=matrix_I
#         # 获取邻接矩阵行累加向量
#         sum_row=tf.expand_dims(tf.reduce_sum(adj,1),1)
#         # 将邻接矩阵做归一化处理
#         adj/=sum_row
#         # 对输入进行计算
#         W_X = tf.matmul(self.w,inputs)
#         A_W_X=tf.matmul(W_X,tf.transpose(adj))
#         output=A_W_X+self.b
#         return output



# d1=tf.constant([[0,1],[1,0]],dtype='double')
# # print(d1)
# d=tf.eye(2,dtype='double')

# def call(adj,inputs):
#         w=tf.constant([[2,2],[1,1]],dtype='double')
#         b=tf.constant([0.5],dtype='double')
#         # 创建单位矩阵
#         matrix_I=tf.eye(adj.shape[0],dtype='double')
#          # 邻接矩阵加上自循环
#         adj+=matrix_I
#         # 获取邻接矩阵行累加向量
#         sum_row=tf.expand_dims(tf.reduce_sum(adj,1),1)
#         # 将邻接矩阵做归一化处理
#         adj/=sum_row
#         # 对输入进行计算
#         W_X = tf.matmul(w,inputs)
#         A_W_X=tf.matmul(W_X,tf.transpose(adj))
#         output=A_W_X+b
#         return output
# print(call(d1,d))

# # print(d1.shape[1])
# # d2=tf.constant([3,4])
# # d1=tf.transpose(d1)
# # d1/=d2
# # print(tf.transpose(d1))

# print(tf.ones([10]))

# 测试张量拼接
# v1=[]
# v1.append(tf.constant([[1,2,3],[1,2,1]]))
# v1.append(tf.constant([[1,1,1]]))
# print(v1)
# v1=tf.concat(v1,0)
# print(v1)
# v1.append(tf.constant([[1,1,1]]))

# print(tf.concat(v1,0))
# print(v1)

bantch1=torch.Tensor([
    [
        [1,1,1],[1,1,1],[1,1,1]
    ],
    [
        [2,2,2],[2,2,2],[2,2,2]
    ]
])
print(bantch1.size())
bantch2=torch.Tensor([
    [
        [2,2,3],[2,2,2],[2,2,2]
    ],
    [
        [2,2,3],[2,2,2],[2,2,2]
    ]
])
print(bantch2.size())
lis = []
lis.append(bantch1)
lis.append(bantch2)
v=torch.cat(lis,dim=1)
print(v)
