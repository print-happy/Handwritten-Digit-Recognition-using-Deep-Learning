import torch
from torch import nn
from d2l import torch as d2l

def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    num_epochs = 10
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, loss {l:f}')

    with torch.no_grad():
        correct, total = 0, 0
        for X, y in test_iter:
            correct += (net(X).argmax(dim=1) == y).type(torch.float).sum().item()
            total += y.shape[0]
        print(f'accuracy: {correct / total:f}')

if __name__ == '__main__':
    main()