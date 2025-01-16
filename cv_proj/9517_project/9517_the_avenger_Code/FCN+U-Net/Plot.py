import matplotlib.pyplot as plt
import random

random_number = random.random()

def plot_loss(train_loss, val_loss, save_path='loss_plot.png'):
    plt.figure(figsize=(10, 6))

    plt.plot(train_loss, label='FCN', color='blue', linestyle='-', marker='o')

    plt.plot(val_loss, label='FCN+ASPP', color='red', linestyle='-', marker='x')

    plt.title('Train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)

    #plt.savefig(save_path)

    plt.show()

train_loss = [1.303, 0.7876, 0.5844, 0.5412, 0.5162, 0.4997, 0.4801
              ]

val_loss = [0.9139, 0.5673, 0.4669, 0.3842, 0.3473, 0.3297, 0.3181
              ]


plot_loss(train_loss, val_loss, 'loss_curve.png')
