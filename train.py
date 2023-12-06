import os
import torch

from logger import logger
from config import device, with_lrs_checkpoint_save_dir, without_lrs_checkpoint_save_dir
from utils import load_checkpoint, save_checkpoint

def train_model(model, criterion, optimizer, train_loader, valid_loader, scheduler=None, num_epochs=1, start_epoch=0, use_checkpoint=False, checkpoint_path=None):
    # May resume from specified checkpoint
    if use_checkpoint and checkpoint_path:
        model, optimizer, start_epoch, _, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info("Resuming from checkpoint")
    
    checkpoint_save_dir = with_lrs_checkpoint_save_dir if scheduler else without_lrs_checkpoint_save_dir

    # Lists for saving metrics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log batch-level statistics
            batch_loss = loss.item()
            batch_acc = (predicted == labels).sum().item() / labels.size(0)
            train_loss_history.append(batch_loss)
            train_acc_history.append(batch_acc)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch+1} finished, Training Loss: {epoch_loss}, Training Accuracy: {epoch_acc}")

        # Validation Phase
        model.eval()
        val_running_loss = 0
        val_total = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            # I haven't found an efficient way to save validation metrics at batch time       
    
        val_epoch_loss = val_running_loss / len(valid_loader)
        val_epoch_acc = val_correct / val_total
        logger.info(f"Epoch {epoch+1} finished, Validation Loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_acc}")
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        save_checkpoint({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": train_loss_history,
            "training_accuracy": train_acc_history,
            "validation_loss": val_loss_history,
            "validation_accuracy": val_acc_history,
        }, filename=os.path.join(checkpoint_save_dir, "epoch_X.pth".replace("X", str(epoch+1))))

        # One model doesn"t have a learning rate scheduler
        if scheduler:
            scheduler.step()
    
    logger.info("Training complete")
