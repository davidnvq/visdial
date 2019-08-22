import torch
import torch.nn as nn
import time

net = nn.Sequential(
    *[nn.Linear(4096, 4096)]*100,
)

net = net.cuda()
net = nn.DataParallel(net)

for i in range(100000000):
    x = torch.randn(12000, 4096)
    y = net(x)
    time.sleep(1)
    if i % 10000 == 0:
        print(i, y.shape)