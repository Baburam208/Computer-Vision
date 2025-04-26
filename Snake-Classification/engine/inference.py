# Inference/prediction code

import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import yaml


class Predictor:
    def __init__(self, model_path, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # If config is not provided, try to get it from checkpoint
        if self.config is None and 'config' in checkpoint:
            self.config = checkpoint['config']

        if self.config is None:
            raise ValueError("Model configuration not provided and not found in checkpoint.")

        # Load model architecture
        from models.pretrained import load_pretrained_model
        model = load_pretrained_model(
            self.config['model']['name'],
            self.config['model']['num_classes'],
            freeze_features=False
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model

    def predict(self, image_path, return_probabilities=False):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        if return_probabilities:
            return predicted.item(), probabilities.cpu().numpy()[0]
        return predicted.item()

    def predict_batch(self, image_paths, return_probabilities=False):
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            images.append(image)

        image_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        if return_probabilities:
            return predicted.cpu().numpy(), probabilities.cpu().numpy()
        return predicted.cpu().numpy()
