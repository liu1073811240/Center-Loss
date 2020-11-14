import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from Net_Model import Net
from center_loss import center_loss
import os
import numpy as np

if __name__ == '__main__':
    save_path = "models/net_center.pth"
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    train_data = torchvision.datasets.MNIST(root="./MNIST", download=True, train=True,
                                            transform=transforms)
    test_data = torchvision.datasets.MNIST(root="./MNIST", download=True, train=False,
                                           transform=transforms)
    train_loader = data.DataLoader(dataset=train_data, shuffle=True, batch_size=512,
                                   num_workers=2)
    test_loader = data.DataLoader(dataset=test_data, shuffle=True, batch_size=256,
                                  num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Param")

    'CrossEntropyLoss()=torch.log(torch.softmax(None))+nn.NLLLoss()'
    'CrossEntropyLoss()=log_softmax() + NLLLoss() '
    'nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合'

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.NLLLoss()
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0)  # 前面10轮动量0.9，中间十轮动量0.3， 后面十轮动量为0
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

    for epoch in range(100000):
        feat_loader = []
        label_loader = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            feature, output = net.forward(x)

            # print(feature.shape)  # torch.Size([100, 2])
            # print(output.shape)  # torch.Size([100, 10])

            loss_cls = loss_fn(output, y)  # output已经用log_softmax输出， 损失函数为NLLLoss
            y = y.float()

            loss_center = center_loss(feature, y, 0.5)  # 比重2可以给小一些，比如0.5

            loss = loss_cls + loss_center  # CELoss(相当于softmax_loss) + Center loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(y.shape)  # torch.Size([100])
            feat_loader.append(feature)
            label_loader.append(y)

            if i % 10 == 0:
                print("epoch:", epoch, "i:", i, "total_loss:", loss.item(),
                      "Softmax_loss", loss_cls.item(), "center_loss", loss_center.item())

        feat = torch.cat(feat_loader, 0)
        labels = torch.cat(label_loader, 0)
        # print(feat)
        # print(labels)
        # print(feat.shape)  # torch.Size([60000, 2])
        # print(labels.shape)  # torch.Size([60000])
        net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
        torch.save(net.state_dict(), save_path)

        eval_loss_cls = 0
        eval_acc_cls = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            feature, output = net.forward(x)

            loss_cls = loss_fn(output, y)
            y_f = y.float()
            loss_center = center_loss(feature, y_f, 2)

            loss = loss_cls + loss_center

            eval_loss_cls += loss_cls.item() * y.size(0)
            out_argmax = torch.argmax(output, 1)
            eval_acc_cls += (out_argmax == y).sum().item()

        mean_loss_cls = eval_loss_cls / len(test_data)
        mean_acc_cls = eval_acc_cls / len(test_data)
        print("分类平均损失：{} 分类平均精度{}".format(mean_loss_cls, mean_acc_cls))

        # 分类问题用精度判断，
        # 1.训练完以后，改进网络模型、用不同的优化器去优化。（把centerloss写成一个类。中心点是可训练的。）
        # 2.SGD学习率可以改为0.5













