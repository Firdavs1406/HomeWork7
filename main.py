import torch
from src.data.make_dataset import load_and_preprocess_data, split_data, prepare_embeddings
from src.model.LSTMNET import LSTMNet
from src.train import train_model
from src.visualization.plot_results import plot_history
from torch.utils.data import DataLoader
from src.model.LSTMNET import FastTextDataset
import torch.optim as optim
import yaml

with open("cfg/train_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Parameters for modeling and training
EMB_DIM = config['train_params']['emb_dim']
NUM_HIDDEN = config['train_params']['num_hidden_nodes']
NUM_OUTPUT = config['train_params']['num_output_nodes']
NUM_LAYERS = config['train_params']['num_layer']
BIDIRECTION = config['train_params']['bidirection']
DROPOUT = config['train_params']['dropout']
BATCH_SIZE = config['train_params']['batch_size']
NUM_EPOCHS = config['train_params']['num_epochs']
LEARNING_RATE = config['train_params']['learning_rate']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Data loading and preprocessing
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Embedding preparation
    X_train_emb, X_test_emb, emb_dict = prepare_embeddings(X_train, X_test, EMB_DIM)

    # Creating datasets and data loaders
    train_dataset = FastTextDataset(X_train_emb, y_train)
    test_dataset = FastTextDataset(X_test_emb, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialization of model, optimizer and loss function
    model = LSTMNet(EMB_DIM, NUM_HIDDEN, NUM_OUTPUT, NUM_LAYERS, BIDIRECTION, DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss().to(DEVICE)

    # Model training
    print("Beginning of training...")
    train_loss_hist, test_loss_hist = train_model(
        model, optimizer, criterion, train_dataloader, test_dataloader, NUM_EPOCHS, DEVICE
    )

    # Visualization of results
    print("Visualization of results...")
    plot_history(train_loss_hist, test_loss_hist)

    print("Training completed.")


if __name__ == "__main__":
    main()
