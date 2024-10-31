import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

logger.add(
    "logs/training.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train_model(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    num_epochs,
    device
):
    train_loss_hist = []
    test_loss_hist = []
    logger.info("Starting training process")

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch, train_acc_epoch = [], []

        for X_batch, y_batch, text_lengths in tqdm(train_dataloader):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output_train = model(X_batch, text_lengths).squeeze()
            loss_train = criterion(output_train, y_batch)
            loss_train.backward()
            optimizer.step()

            train_loss_epoch.append(loss_train.item())
            train_acc_epoch.append(binary_accuracy(output_train, y_batch).item())

        train_loss = np.mean(train_loss_epoch)
        train_acc = np.mean(train_acc_epoch)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
        )

        model.eval()
        test_loss_epoch = []

        with torch.no_grad():
            for X_batch, y_batch, text_lengths in test_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output_test = model(X_batch, text_lengths).squeeze()
                loss_test = criterion(output_test, y_batch)
                test_loss_epoch.append(loss_test.item())

        test_loss = np.mean(test_loss_epoch)
        test_loss_hist.append(test_loss)
        train_loss_hist.append(train_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Test Loss: {test_loss:.4f}")

    logger.info("Training process completed")
    return train_loss_hist, test_loss_hist
