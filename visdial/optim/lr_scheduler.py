import math
from bisect import bisect


class LRScheduler(object):

    def __init__(self, optimizer,
                 batch_size, num_samples, num_epochs,
                 init_lr=0.01,
                 min_lr=1e-4,
                 warmup_factor=0.1,
                 warmup_epochs=1,
                 scheduler_type='CosineLR',
                 milestone_steps=[4, 8, 12, 16, 20, 24, 28],
                 linear_gama=0.5,
                 **kwargs
                 ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        self._SCHEDULER = {
            'CosineLR': self.cosine_step,
            'LinearLR': self.linear_step
        }
        self.scheduler_step = self._SCHEDULER[scheduler_type]
        self.scheduler_type = scheduler_type

        self.batch_size = batch_size
        self.num_samples = num_samples

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.num_epochs = num_epochs

        if num_samples % batch_size == 0:
            self.total_iters_per_epoch = num_samples // batch_size
        else:
            self.total_iters_per_epoch = num_samples // batch_size + 1

        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs

        self.linear_gama = linear_gama
        self.milestone_steps = milestone_steps

    def step(self, cur_iter):
        current_epoch = cur_iter / self.total_iters_per_epoch
        if current_epoch < 1:
            lr = self.warmup_step(cur_iter)
        else:
            lr = self.scheduler_step(cur_iter)

        lr = self.min_lr if lr < self.min_lr else lr
        self.update(lr)
        return lr

    def update(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_step(self, cur_iter):
        """
        current_iteration: 1 iter = 1 batch step
        (should be understood as the global_steps accumulated from the beginning.)
        return the factor.
        """
        current_epoch = float(cur_iter) / float(self.total_iters_per_epoch)
        alpha = current_epoch / self.warmup_epochs
        return self.init_lr * (self.warmup_factor * (1.0 - alpha) + alpha)

    def linear_step(self, cur_iter):
        current_epoch = cur_iter // self.total_iters_per_epoch
        idx = bisect(self.milestone_steps, current_epoch)
        return self.init_lr * pow(self.linear_gama, idx)

    def cosine_step(self, cur_iter):
        # current_epoch = cur_iter // self.total_iters_per_epoch
        # return self.init_lr * (1 + math.cos(math.pi * current_epoch / self.num_epochs)) / 2
        total_iters = self.num_epochs * self.total_iters_per_epoch
        return self.init_lr * (1 + math.cos(math.pi * cur_iter / total_iters)) / 2


def test_lr_scheduler(scheduler_type='LinearLR'):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    adam = torch.optim.Adam(torch.nn.Linear(2, 3).parameters(), lr=0.01)
    num_epochs = 30
    num_iter_per_epoch = 10
    batch_size = 8
    num_samples = 80
    init_lr = 0.01
    min_lr = 1e-5
    scheduler = LRScheduler(adam, batch_size,
                            num_samples, num_epochs,
                            scheduler_type=scheduler_type,
                            init_lr=init_lr, min_lr=min_lr)

    lr = []
    global_steps = 0
    for epoch in range(num_epochs):
        for i in range(num_iter_per_epoch):
            lr.append(scheduler.step(global_steps))
            # print(adam.state_dict)
            global_steps += 1


    for i, l in enumerate(lr):
        print("%.5f" % l, end=' ')
        if (i+1) % num_iter_per_epoch == 0:
            print("")

    plt.plot(np.arange(global_steps), lr)
    plt.ylim(0, init_lr)
    plt.show()