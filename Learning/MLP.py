import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

net=nn.Sequential(nn.Flatten()
                  ,nn.Linear(784,256)
                  ,nn.ReLU()
                  ,nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def evaluate_accuracy(net, data_iter):
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

accuracy = evaluate_accuracy(net, test_iter)
print(f'Accuracy: {accuracy:.2f}')