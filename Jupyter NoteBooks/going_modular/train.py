"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import torch
import os
from torchvision import transforms
import data_setup
import utils
import engine
import model_builder
from timeit import default_timer as timer

# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup Hyper parameters
learning_rate = 0.001
num_epochs = 3
batch_size = 32
hidden_units = 10
input_shape = 3


# Setup directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

# Device agnostice code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformmation
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                            test_dir=test_dir,
                                                            transform=data_transform,
                                                            batch_size=batch_size)

# Create model
model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = timer()

# Train model with engine
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             device=device,
             epochs=num_epochs
             )

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model using utils
utils.save_model(model=model,
                 model_name='06_going_modular_script_mode.pth',
                 target_dir='models')
