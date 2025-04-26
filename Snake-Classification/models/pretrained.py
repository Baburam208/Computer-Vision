import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights, ResNet50_Weights, DenseNet121_Weights


def load_pretrained_model(model_name, num_classes, freeze_features=True):
    """Load pretrained model and modify for custom task"""
    model_dict = {
        'vgg16': (models.vgg16, VGG16_Weights.IMAGENET1K_V1),
        'resnet50': (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
        'densenet121': (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1)
    }

    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported")

    model_fn, weights_enum = model_dict[model_name]
    model = model_fn(weights=weights_enum)

    # Freeze feature extractor
    if freeze_features:
        if model_name == 'vgg16':
            for param in model.features.parameters():
                param.require_grad = False
        elif model_name == 'resnet50':
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
            for param in model.layer3.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = False
        elif model_name == 'densenet121':
            for param in model.features.parameters():
                param.requires_grad = False

    # Modify classifier/head
    if model_name == 'vgg16':
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(512),  # Add batch norm
            nn.Linear(in_features=512, out_features=num_classes)
        )
    elif model_name == 'resnet50':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model
