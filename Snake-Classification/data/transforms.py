# Custom transforms
from torchvision import transforms


def get_train_transforms():
    return transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),  # lighting and color variation
        transforms.RandomAffine(
            degrees=20,  # rotation_range
            translate=(0.1, 0.1),  # width_shift_range & height_shift_range
            scale=(0.8, 1.2),  # zoom_range: (1-0.2, 1+0.2)
            shear=0.2,  # shear_range
            fill=0  # fill with 0 (black) instead of fill_mode='nearest' ('nearest' not directly available in pytorch)
        ),
        transforms.RandomGrayscale(p=0.1),  # Randomly convert images to grayscale
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # Mimics blur from motion or focus issues
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
