# RNN Binary Classification Project

## Overview
This project implements RNN-based model - LSTM for binary text classification tasks using word embedding approaches.

## Features
- LSTM architecture
- Word embedding option - FastText
- Configurable hyperparameters
- Comprehensive logging
- Test coverage

## Project Structure
```
.
├── data/           # Data directory
├── src/            # Source code
├── logs/           # Training logs
├── cfg/            # Configuration files
```

## Installation

### Using Makefile
```bash
make setup
```

## Usage

### Run
```bash
# Using default configuration
make run
```

### Code Quality
```bash
# Run linters and formatters
make check
```

## Configuration
The model and training parameters can be configured in `cfg/train_config.yaml`:
- Learning rate
- Batch size
- Number of epochs
- Model architecture
- Embedding type
- etc.

## Logging
Training progress and metrics are logged to:
- Console output
- `logs/training.log`
