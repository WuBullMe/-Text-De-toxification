import matplotlib.pyplot as plt

def plot_loss(loss_train, loss_val, epochs):
    plt.plot(range(1, epochs + 1), loss_train, label='Training loss')
    plt.plot(range(1, epochs + 1), loss_val, label='Validation loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend(loc='best')
    plt.show()