# library
# standard library
from matplotlib import cm
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 包刮了一些數據庫跟圖片的數據庫
import matplotlib.pyplot as plt
import time

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False  # 下載了寫False，還沒下載寫True


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

# 下載minist data
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data 會有六萬個數據點，false的話只會有一萬
    # Converts a PIL.Image or numpy.ndarray to
    #
    transform=torchvision.transforms.ToTensor(),
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    # 要怎麼把網路上下載的原始數據，改變成你要的形式，這裡就是把她轉成tensor的形式
    # 把pixel轉成tensor
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing, so train = false, if = true is all data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
    :2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


# 建立CNN網路


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28) #卷積層
            # like filter 三維，有多少個filter就會提取多高或是多少種類的特徵訊息
            nn.Conv2d(
                in_channels=1,              # input height 第一層只會是1
                out_channels=16,            # n_filters 同時有16個filter疊在一起，同時掃描某一個區域，紀錄16個特徵
                kernel_size=5,              # filter size 5*5的區域大小一個一個掃描
                stride=1,                   # filter movement/step 每掃描一次，只跳一個pixel
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                padding=2,  # 把圖片外圍包一層0
            ),                              # output shape (16, 28, 28) 高度變高
            nn.ReLU(),                      # activation
            # choose max value in 2x2 area, output shape (16, 14, 14) 大小便小
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        # fully connected layer, output 10 classes 因為0~9
        self.out = nn.Linear(32 * 7 * 7, 10)

    # 把上面三維的數據展平成二維的數據
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch_size, 32, 7, 7)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)  # x.size(0)保留維度, -1把後面都乘在一起
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

# optimize all cnn parameters
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# the target label is not one-hotted
loss_func = nn.CrossEntropyLoss()

# following function (plot_with_labels) is for visualization, can be ignored if not interested
try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except:
    HAS_SK = False
    print('Please install sklearn for layer visualization')


# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9))
#         plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max())
#     plt.ylim(Y.min(), Y.max())
#     plt.title('Visualize last layer')
#     plt.show()
#     plt.pause(0.01)


# plt.ion()
t1 = time.time()  # time start
# training and testing
for epoch in range(EPOCH):
    # gives batch data, normalize x when iterate train_loader
    for step, (b_x, b_y) in enumerate(train_loader):

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(
                int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' %
                  loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2,
                            init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(
                    last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                # plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

t2 = time.time()  # time end
print('time elapsed: ' + str(t2-t1) + ' seconds')
