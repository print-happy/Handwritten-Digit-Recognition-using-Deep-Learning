import os
import struct
from array import array

# 定义一个MNIST类
class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        # 测试集图像和标签文件名
        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        # 训练集图像和标签文件名
        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        # 初始化测试集图像和标签列表
        self.test_images = []
        self.test_labels = []

        # 初始化训练集图像和标签列表
        self.train_images = []
        self.train_labels = []

    # 加载测试集数据
    def load_testing(self):
        # 调用load方法加载测试集图像和标签
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        # 将加载的图像和标签赋值给测试集图像和标签列表
        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    # 加载训练集数据
    def load_training(self):
        # 调用load方法加载训练集图像和标签
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        # 将加载的图像和标签赋值给训练集图像和标签列表
        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    # 静态方法，用于加载图像和标签数据
    @classmethod
    def load(cls, path_img, path_lbl):
        # 打开标签文件
        with open(path_lbl, 'rb') as file:
            # 读取文件头部的魔数和数据大小
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            # 读取标签数据
            labels = array("B", file.read())

        # 打开图像文件
        with open(path_img, 'rb') as file:
            # 读取文件头部的魔数、数据大小、行数和列数
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            # 读取图像数据
            image_data = array("B", file.read())

        # 初始化图像列表
        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        # 将图像数据按行列顺序存储到图像列表中
        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    # 静态方法，用于显示图像
    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render
