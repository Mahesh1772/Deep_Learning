"""
Contains function to save the PyTorch model
"""
import torch
from pathlib import Path

def save_model(model:torch.nn.Module, target_dir:str, model_name:str):
  """ Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should Include either ".pth' or ".pt"

  Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
  """
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(exist_ok=True, parents=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith('.pt'), 'model_name should end with .pth or .pt'
  model_save_path = target_dir_path/model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)
