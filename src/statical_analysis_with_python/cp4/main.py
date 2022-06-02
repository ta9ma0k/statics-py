import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_dice_prob(num_trial: int):
    dice = np.arange(1, 7)
    prob = dice / 21

    dice_sample = np.random.choice(dice, num_trial, p=prob)
    freq, _ = np.histogram(dice_sample, bins=6, range=(1, 7))
    print(pd.DataFrame({'度数': freq, '相対度数': freq / num_trial}, index=pd.Index(np.arange(1, 7), name='出目')))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(dice_sample, bins=6, range=(1, 7), density=True, rwidth=0.8)
    ax.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
    ax.set_xticks(np.linspace(1.5, 6.5, 6))
    ax.set_xticklabels(np.arange(1, 7))
    ax.set_xlabel('出目')
    ax.set_ylabel('相対度数')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../../data/ch4_scores400.csv')
    scores = np.array(df['点数'])

    print(f'母平均={scores.mean()}')
    for i in range(5):
        sample = np.random.choice(scores, 20)
        print(f'{i + 1}回目 標本平均＝{sample.mean()}')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(scores, bins=100, range=(0, 100), density=True)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 0.042)
    ax.set_xlabel('点数')
    ax.set_ylabel('相対度数')
    plt.show()

    sample_mean = [np.random.choice(scores, 20).mean() for _ in range(10000)]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.hist(sample_mean, bins=100, range=(0, 100), density=True)
    ax.vlines(np.mean(scores), 0, 1, 'gray')
    ax.set_xlim(50, 90)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel('点数')
    ax.set_ylabel('相対度数')
    plt.show()