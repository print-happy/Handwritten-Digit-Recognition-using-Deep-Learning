import torch
from torchvision import datasets,transforms
from d2l import torch as d2l

batch_size=256
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_inputs=784
num_outputs=10

W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)

def softmax(X):
    X_exp=torch.exp(X)
    p=X_exp.sum(1,keepdim=True)
    return X_exp/p

def net(X):
    return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:  
        y_hat=y_hat.argmax(axis=1)  #将y_hat化成行向量，其中每一位表示对应样本第几位概率最大
    cmp=y_hat.type(y.dtype)==y  #用bool向量储存预测准确的样本
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data=[0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.reshape(-1).size(0))
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        print(epoch+1," ",test_acc,"\n")
    
lr=0.1

def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)

if __name__ == '__main__':
    num_epochs=10
    train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)