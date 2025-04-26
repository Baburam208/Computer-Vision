import torch
import os
from tqdm import tqdm
from datetime import datetime
from utils.logger import TensorBoardLogger
from utils.metrics import MetricTracker


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.previous_lr = optimizer.param_groups[0]['lr']  # Initialize with starting LR Scheduler
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.early_stop = True

        # Metrics
        self.train_metrics = MetricTracker(config['model']['num_classes']).to(device)
        self.val_metrics = MetricTracker(config['model']['num_classes']).to(device)

        # Logger
        self.logger = TensorBoardLogger(
            config['paths']['log_dir'],
            experiment_name=f"{config['model']['name']}_train"
        )

        # Training state
        self.best_val_accuracy = 0.0
        self.current_epoch = 0

        # Early stopping parameters
        self.early_stop = config['training'].get('early_stop', False)
        self.patience = config['training'].get('es_patience', 5)
        self.delta = config['training'].get('delta', 0.001)
        self.counter = 0
        self.best_metric = -float('inf')  # Initialize to negative infinity for accuracy tracking
        self.early_stop_triggered = False

    def _check_early_stopping(self, val_accuracy):
        """Check if early stopping conditions are met"""
        if not self.early_stop:
            return False

        if val_accuracy > self.best_metric + self.delta:
            self.best_metric = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop_triggered = True
                return True
        return False

    def _update_learning_rate_logging(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.log_scalar('Epoch/Learning Rate', current_lr, self.current_epoch)

        if hasattr(self, 'previous_lr') and current_lr != self.previous_lr:
            print(f"LR changed from {self.previous_lr:.2e} to {current_lr:.2e}")

        self.previous_lr = current_lr

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        self.train_metrics.reset()

        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            self.train_metrics.update(preds, labels)

            # Log batch metrics
            self.logger.log_scalar(
                tag="Batch/Train Loss",
                value=loss.item(),
                step=self.current_epoch * len(self.train_loader) + batch_idx
            )

        epoch_loss = running_loss / len(self.train_loader.dataset)
        train_metrics = self.train_metrics.compute()

        return {
            "loss": epoch_loss,
            "accuracy": train_metrics['accuracy'],
            "f1": train_metrics['f1']
        }

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        self.val_metrics.reset()

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                self.val_metrics.update(preds, labels)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        val_metrics = self.val_metrics.compute()

        return {
            "loss": epoch_loss,
            "accuracy": val_metrics['accuracy'],
            "f1": val_metrics['f1']
        }

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'early_stop_triggered': self.early_stop_triggered
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = f"{self.config['model']['name']}_best_{timestamp}.pth"
        else:
            filename = f"{self.config['model']['name']}_checkpoint_{timestamp}.pth"

        save_path = os.path.join(self.config['paths']['model_save_dir'], filename)
        torch.save(state, save_path)
        return save_path

    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train and validate
            train_results = self.train_epoch()
            val_results = self.validate()

            # Update learning rate
            self.scheduler.step(val_results['loss'])

            # Check for early stopping
            if self._check_early_stopping(val_results['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            # Log the learning rate
            # current_lr = self.optimizer.param_groups[0]['lr']
            # self.logger.log_scalar('Epoch/Learning Rate', current_lr, epoch)
            # if current_lr != self.previous_lr:
            #     print(f"Learning rate reduced to {current_lr}")
            # self.previous_lr = current_lr
            self._update_learning_rate_logging()

            # Log epoch metrics
            self.logger.log_scalars(
                "Epoch/Accuracy",
                {"Train": train_results['accuracy'], "val": val_results['accuracy']},
                epoch
            )
            self.logger.log_scalars(
                "Epoch/F1",
                {"Train": train_results['f1'], "Val": val_results['f1']},
                epoch
            )
            self.logger.log_scalars(
                "Epoch/Loss",
                {"Train": train_results['loss'], "Val": val_results["loss"]},
                epoch
            )
            self.logger.log_scalar(
                "Epoch/Learning Rate",
                self.optimizer.param_groups[0]['lr'],
                epoch
            )

            # Save best model
            if val_results['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_results['accuracy']
                save_path = self.save_checkpoint(is_best=True)
                print(f"\nNew best model saved at {save_path} with val accuracy: {self.best_val_accuracy:.4f}")

            # Print progress
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_results['loss']:.4f} | Accuracy: {train_results['accuracy']:.4f} | F1: {train_results['f1']:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f} | Accuracy: {val_results['accuracy']:.4f} | F1: {val_results['f1']:.4f}")
            print()

        self.logger.close()
        return self.best_val_accuracy
