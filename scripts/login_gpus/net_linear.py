import torch
import torch.nn as nn

net = nn.Sequential(
    *[nn.Linear(2048, 2048)]*100,
)

net = net.cuda()
net = nn.DataParallel(net)

for i in range(1000000):
    x = torch.randn(512, 2048)
    y = net(x)