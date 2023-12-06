from matplotlib import pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from logger import logger
from config import device, learning_rate, use_checkpoint, num_epochs, with_lrs_checkpoint_path, without_lrs_checkpoint_path
from data import train_loader, valid_loader, num_classes
from models import MaizeLeafCNN
from train import train_model
from utils import predict_image, load_checkpoint, simple_moving_average

logger.info("Instantiating models")

# Model with a learning rate scheduler
model1 = MaizeLeafCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer1 = Adam(model1.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer1, step_size=30, gamma=0.5)

logger.info("Starting training the model with a scheduler")
train_model(
    model1, 
    criterion, 
    optimizer1, 
    train_loader, 
    valid_loader,
    scheduler=scheduler,
    num_epochs=num_epochs,
    use_checkpoint=use_checkpoint, 
    checkpoint_path=with_lrs_checkpoint_path
)

# Model without a learning rate scheduler
model2 = MaizeLeafCNN(num_classes).to(device)
optimizer2 = Adam(model2.parameters(), lr=learning_rate)

logger.info("Starting training the model without a scheduler")
train_model(
    model2, 
    criterion, 
    optimizer2, 
    train_loader, 
    valid_loader, 
    num_epochs=num_epochs,
    use_checkpoint=use_checkpoint, 
    checkpoint_path=without_lrs_checkpoint_path
)

# Results
logger.info("Plotting the results")
epochs = range(1, num_epochs + 1)
_, _, _, training_loss1, training_accuracy1, validation_loss1, validation_accuracy1 = load_checkpoint(with_lrs_checkpoint_path, model1, optimizer1)
_, _, _, training_loss2, training_accuracy2, validation_loss2, validation_accuracy2 = load_checkpoint(without_lrs_checkpoint_path, model2, optimizer2)

# Smoothing
training_loss1 = simple_moving_average(training_loss1)
training_accuracy1 = simple_moving_average(training_accuracy1)
validation_loss1 = simple_moving_average(validation_loss1)
validation_accuracy1 = simple_moving_average(validation_accuracy1)

training_loss2 = simple_moving_average(training_loss2)
training_accuracy2 = simple_moving_average(training_accuracy2)
validation_loss2 = simple_moving_average(validation_loss2)
validation_accuracy2 = simple_moving_average(validation_accuracy2)

plt.figure(figsize=(12, 6))
plt.suptitle("Model comparaison", fontsize=16)

# Subplots
plt.subplot(2, 2, 1)
plt.plot(training_loss1, label="Training loss")
plt.plot(validation_loss1, label="Validation loss")
plt.title("Model with LRS ⬇️")
plt.xlabel("Époques")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(training_loss2, label="Training loss")
plt.plot(validation_loss2, label="Validation loss")
plt.title("Model without LRS ⬇️")
plt.xlabel("Époques")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(training_accuracy1, label="Training accuracy")
plt.plot(validation_accuracy1, label="Validation accuracy")
plt.xlabel("Époques")
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(training_accuracy2, label="Training accuracy")
plt.plot(validation_accuracy2, label="Validation accuracy")
plt.xlabel("Époques")
plt.ylim(0, 1)
plt.legend()

plt.subplots_adjust(top=0.82, bottom=0.1, left=0.08, right=0.92, hspace=0.3, wspace=0.3)
plt.show()

# ### TESTS ###
# import os
# from PIL import Image

# class_names = train_data.classes

# for img in os.listdir(test_dir):
#     img_path = os.path.join(test_dir, img)
#     image = Image.open(img_path)
#     image_tensor = transform(image)
#     predicted_class_index = predict_image(image_tensor, model, device)
#     predicted_class_name = class_names[predicted_class_index]
#     print(f"{img} : Predicted Class: {predicted_class_name}")

# image_path = "test/TomatoHealthy2.JPG"
# predicted_class_index = predict_image(image_path, model, transform)
# class_names = train_data.classes
# predicted_class_name = class_names[predicted_class_index]
# print(f"Predicted Class: {predicted_class_name}")
