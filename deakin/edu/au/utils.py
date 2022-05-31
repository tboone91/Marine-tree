# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt
import numpy as np


def learning_plots(history):
    plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(1, 2, 1)
    for l in history.history:
        if l == 'loss' or l == 'val_loss':
            loss = history.history[l]
            plt.plot(range(1, len(loss) + 1), loss, label=l)

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    for k in history.history:
        if 'accuracy' in k:
            loss = history.history[k]
            plt.plot(range(1, len(loss) + 1), loss, label=k)
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def get_potential_improvement(true_y, pred_y, taxo):
    pred_y = [np.argmax(x, axis=1) for x in pred_y]
    oracle_accuracy1 = [0] * len(pred_y)
    oracle_accuracy2 = [0] * len(pred_y)
    oracle_accuracy = [0] * len(pred_y)
    for i in range(len(pred_y)):
        for j in range(len(pred_y[0])):
            # check correctness
            if pred_y[i][j] == true_y[i][j][0]:
                continue
            else:
                # go left to right
                stop = False
                for z in range(i + 1, len(pred_y)):
                    if pred_y[z][j] == true_y[z][j][0]:
                        oracle_accuracy[i] = oracle_accuracy[i] + 1
                        oracle_accuracy1[i] = oracle_accuracy1[i] + 1
                        stop = True
                        break
                # go right to left -- do not touch, complex code
                if stop == False and i > 0:
                    parents = []
                    current = pred_y[i][j]
                    for z in reversed(range(i)):
                        m = taxo[z]
                        row = list(np.transpose(m)[current])
                        parent = row.index(1)
                        current = parent
                        parents.insert(0, parent)
                    for z in reversed(range(i)):
                        if pred_y[z][j] == true_y[z][j][0] and true_y[z][j][0] != parents[z]:
                            oracle_accuracy[i] = oracle_accuracy[i] + 1
                            oracle_accuracy2[i] = oracle_accuracy2[i] + 1
                            break

    print("Total improvement: ", oracle_accuracy)
    print("Left to right improvement: ", oracle_accuracy1)
    print("Right to left improvement: ", oracle_accuracy2)
    oracle_accuracy = [x * 100 / len(true_y[0]) for x in oracle_accuracy]
    return oracle_accuracy


def plot_potential_improvement(oracle_accuracy, model_accuracy):
    labels = ['Level 0', 'Level 1', 'Level 2']
    width = 0.6
    params = {'legend.fontsize': 10,
              'axes.labelsize': 18,
              'axes.titlesize': 11,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.titlepad': 12}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_axisbelow(True)
    ax.bar(labels, model_accuracy, width, edgecolor='black', linewidth=0.5,
           color=['#FF3400', '#0025FF', '#008000', '#FFFD07'])
    ax.bar(labels, oracle_accuracy, width, bottom=model_accuracy, edgecolor='black', linewidth=0.5,
           color=['#FFC0CB', '#ADD8E6', '#90EE90', '#FFFEE0'])
    ax.set_ylabel('Fraction %')
    rects = ax.patches
    labels = ["+%.2f" % i for i in oracle_accuracy]
    labels = [x + "%" for x in labels]

    heights = []
    for imp, accuracy in zip(oracle_accuracy, model_accuracy):
        heights.append(accuracy + imp / 2)

    for rect, label, height in zip(rects, labels, heights):
        ax.text(rect.get_x() + rect.get_width() / 2, height - 2, label,
                ha='center', va='bottom')

    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax.set_yticks([0,2,4,6])
    ax.set_ylim(0, 100)
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    return plt


if __name__ == '__main__':
    print('')
