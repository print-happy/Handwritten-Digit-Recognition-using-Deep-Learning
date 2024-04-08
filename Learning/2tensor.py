import pandas as pd
import os

data_file = os.path.join('..', 'data', 'programmers.csv')
data = pd.read_csv(data_file)
inputs= data.iloc[:,:3]    #分段
outputs= data.iloc[:,-1]    #分段
inputs.drop(columns=['Programmers'],inplace=True)    #去除姓名行
inputs = inputs.fillna(inputs.mean())    #用平均值填充缺失值
#inputs.mean()返回一个Series，Series的索引是列名，值是平均值，传入fillna用于填充缺失值
inputs = pd.get_dummies(inputs, dummy_na=True)    #将非数值型数据转换为数值型数据，并且使用one_hot编码转化成二进制向量
import torch
torch.set_printoptions(sci_mode=False)    #取消科学计数法（如果后面要用回来得设置为True）
x = torch.tensor(inputs.to_numpy(dtype=float))    #将数据转化为张量
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x,"\n")
print(y)
