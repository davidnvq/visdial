import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import numpy

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize' : (8, 5),
    'axes.labelsize' : 'xx-large',
    'axes.titlesize' : 'x-large',
    'xtick.labelsize': 'xx-Large',
    'ytick.labelsize': 'xx-Large'
}

pylab.rcParams.update(params)



def draw_losses():
    path1 = '/home/quanguet/Downloads/run-.-tag-attn_misc_lstm-train_batch_loss.csv'
    path2 = '/home/quanguet/Downloads/run-.-tag-attn_misc_lstm-train_epoch_loss.csv'

    fig = plt.figure()
    ax = fig.gca()

    df = pd.read_csv(path1)
    ax.plot(df['Step'], df['Value'],
            color='mediumseagreen',
            label='Batch loss')

    df = pd.read_csv(path2)
    ax.plot(df['Step'], df['Value'],
            color='crimson',
            marker='o',
            markersize=4,
            label='Epoch loss'
            )

    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(loc='upper right')

    ax.set_xticks(numpy.arange(0, 90000, 20000))
    ax.set_xticklabels([0, '2k', '4k', '6k', '8k'])

    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


paths = [
    '/home/quanguet/Downloads/run-attn_misc_lstm-train_metrics_r@1-tag-attn_misc_lstm-train_metrics.csv',
    '/home/quanguet/Downloads/run-attn_misc_lstm-train_metrics_r@5-tag-attn_misc_lstm-train_metrics.csv',
    '/home/quanguet/Downloads/run-attn_misc_lstm-train_metrics_r@10-tag-attn_misc_lstm-train_metrics.csv',
    '/home/quanguet/Downloads/run-attn_misc_lstm-train_metrics_mrr-tag-attn_misc_lstm-train_metrics.csv',
]


def draw_metrics_plot(paths):
    fig = plt.figure()
    ax = fig.gca()

    labels = ['R@1', 'R@5', 'R@10', 'MRR']
    colors = ['red', 'green', 'blue', 'black']
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        ax.plot(df['Step'], df['Value'],
                color=colors[i],
                label=labels[i],
                # marker='o',
                markersize=3)

    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(loc='lower right')

    ax.set_xticks(numpy.arange(0, 100000, 20000))
    ax.set_xticklabels(['0', '20k', '40k', '60k', '80k'])

    plt.grid(True, linestyle=':')
    plt.show()
    fig.savefig("foo.pdf", bbox_inches='tight')
