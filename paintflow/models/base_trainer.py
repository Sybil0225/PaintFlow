import os
import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base trainer providing reusable structure for training pipelines.
    Subclasses should implement specific training and validation logic.
    """

    def __init__(self, model, optimizer, scheduler, train_dataloader, val_dataloader, device="cuda"):
        self.model = model.to(device)  # Model to be trained
        self.optimizer = optimizer  # Optimizer
        self.scheduler = scheduler  # Learning rate scheduler
        self.train_dataloader = train_dataloader  # Training data loader
        self.val_dataloader = val_dataloader  # Validation data loader
        self.device = device  # Device for training and evaluation
        self.current_epoch = 0  # Current training epoch
        self.best_metric = float("inf")  # Track the best validation metric

    @abstractmethod
    def compute_loss(self, batch):
        """Compute loss for a single batch. To be implemented by subclass."""
        pass

    def train_epoch(self):
        """Perform one training epoch."""
        self.model.train()  # Set the model to training mode
        total_loss = 0

        for step, batch in enumerate(self.train_dataloader):
            # Transfer batch data to the device
            batch = {key: val.to(self.device) for key, val in batch.items()}

            self.optimizer.zero_grad()  # Zero gradients of the optimizer
            loss = self.compute_loss(batch)  # Compute loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate loss

            if step % 50 == 0:
                print(f"Step {step}: Loss = {loss.item()}")

        avg_loss = total_loss / len(self.train_dataloader)
        print(f"Epoch {self.current_epoch}: Training Loss = {avg_loss}")
        return avg_loss

    def validate(self):
        """Perform validation on the validation dataset."""
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                # Transfer batch data to the device
                batch = {key: val.to(self.device) for key, val in batch.items()}

                loss = self.compute_loss(batch)  # Compute validation loss
                total_loss += loss.item()  # Accumulate loss

        avg_loss = total_loss / len(self.val_dataloader)
        print(f"Epoch {self.current_epoch}: Validation Loss = {avg_loss}")
        return avg_loss

    def save_checkpoint(self, output_dir, is_best=False):
        """Save the model and optimizer checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
        }
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")

        torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Best model saved at epoch {self.current_epoch}")

    def train(self, num_epochs, output_dir, save_interval=1):
        """Main training loop."""
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            print(f"Starting epoch {self.current_epoch}...")
            train_loss = self.train_epoch()  # Perform training
            val_loss = self.validate()  # Perform validation

            # Check if validation loss improves
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.save_checkpoint(output_dir, is_best=True)

            # Save checkpoints periodically
            if epoch % save_interval == 0:
                self.save_checkpoint(output_dir)

            # Step the scheduler for learning rate decay
            self.scheduler.step(val_loss)
        print("Training completed!")
