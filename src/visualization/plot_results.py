import matplotlib.pyplot as plt


def plot_history(train_loss_hist, test_loss_hist):
    # Plot the training and test loss histories
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(test_loss_hist, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss History')
    plt.legend()

    # Save the plot to a file
    plt.savefig('src/visualization/loss_history.png')
    plt.close()
