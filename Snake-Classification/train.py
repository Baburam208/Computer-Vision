# Training script
import torch
import yaml
from torch import nn, optim
from pathlib import Path

from data.dataset import get_datasets, get_dataloaders
from models.pretrained import load_pretrained_model
from engine.train import Trainer


def main():
    # Load config
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories
    Path(config['paths']['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)

    # Get data
    train_dataset, val_dataset = get_datasets(
        config['data']['train_path'],
        config['data']['val_path']
    )

    train_loader, val_loader = get_dataloaders(
        train_dataset, val_dataset,
        config['data']['batch_size'],
        config['data']['num_workers']
    )

    # Get model
    model = load_pretrained_model(
        config['model']['name'],
        config['model']['num_classes'],
        freeze_features=True
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters() if config['model']['name'] in ['vgg16', 'densenet121'] else model.fc.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=config['training']['lr_patience'],
        factor=config['training']['factor']
    )

    # Train
    trainer = Trainer(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                config
    )
    best_accuracy = trainer.train()

    print(f"\nTraining complete, Best validation accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
