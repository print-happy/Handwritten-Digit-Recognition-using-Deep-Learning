import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
#from cnn.neural_network import CNN  # 导入CNN类
from cnn.neural_network_ResNet import ResNet

# 解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str)
args = vars(ap.parse_args())

# 加载MNIST数据集
print('加载MNIST数据集...')
dataset = fetch_openml('mnist_784')

# 将MNIST数据转换为28x28图像矩阵
mnist_data = dataset.data.values.reshape((dataset.data.shape[0], 28, 28))

# 为数据添加一个新的轴，表示通道维度
mnist_data = mnist_data[:, np.newaxis, :, :]

# 将数据分为训练集和测试集
train_img, test_img, train_labels, test_labels = train_test_split(
    mnist_data / 255.0, dataset.target.astype("int"), test_size=0.1)

# 设置图像维度
img_rows, img_columns = 28, 28

# 将标签转换为分类格式
total_classes = 10
train_labels = torch.tensor(train_labels.values).long()
test_labels = torch.tensor(test_labels.values).long()

# 定义SGD优化器和CNN模型
import torch.nn as nn

print('\n编译模型...')
#model = CNN(width=img_rows, height=img_columns, depth=1, total_classes=total_classes)
model = ResNet(num_classes=total_classes)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

# 如果已经保存了权重，使用参数加载权重
if args["load_model"] > 0:
	model.load_state_dict(torch.load(args["save_weights"]))

# 训练和测试模型
b_size = 128
num_epoch = 5
losses=[]

if args["load_model"] < 0:
    print('\n训练模型...')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    train_data = TensorDataset(torch.tensor(train_img).float(), train_labels)
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)
    model.train()
    from tqdm import tqdm
    for epoch in range(num_epoch):
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

    print('评估准确性和损失函数...')
    test_data = TensorDataset(torch.tensor(test_img).float(), test_labels)
    test_loader = DataLoader(test_data, batch_size=b_size, shuffle=False)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('模型准确率: {:.2f}%'.format(accuracy * 100))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(losses)
    plt.xlabel('SGD times')
    plt.ylabel('Loss')
    plt.show()



