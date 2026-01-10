import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from app.config import Config

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Charger les images NORMAL (label 0)
        normal_path = os.path.join(root_dir, "NORMAL")
        if os.path.exists(normal_path):
            for img_name in os.listdir(normal_path):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(normal_path, img_name))
                    self.labels.append(0)
        
        # Charger les images PNEUMONIA (label 1)
        pneumonia_path = os.path.join(root_dir, "PNEUMONIA")
        if os.path.exists(pneumonia_path):
            for img_name in os.listdir(pneumonia_path):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(pneumonia_path, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Transformations pour les images"""
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_federated_datasets(dataset_path):
    """
    Divise le dataset en plusieurs hôpitaux pour la simulation fédérée
    """
    train_transform, _ = get_transforms()
    
    # Charger le dataset d'entraînement complet
    train_dataset = ChestXRayDataset(
        os.path.join(dataset_path, "train"),
        transform=train_transform
    )
    
    # Diviser en N hôpitaux
    total_size = len(train_dataset)
    hospital_size = total_size // Config.NUM_HOSPITALS
    sizes = [hospital_size] * Config.NUM_HOSPITALS
    sizes[-1] += total_size - sum(sizes)  # Ajouter le reste au dernier hôpital
    
    hospital_datasets = random_split(
        train_dataset, 
        sizes,
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )
    
    return hospital_datasets

def get_dataloaders(dataset_path):
    """
    Crée les dataloaders pour l'entraînement et le test
    """
    train_transform, test_transform = get_transforms()
    
    # Dataset de test
    test_dataset = ChestXRayDataset(
        os.path.join(dataset_path, "test"),
        transform=test_transform
    )
    
    # Dataset de validation
    val_dataset = ChestXRayDataset(
        os.path.join(dataset_path, "val"),
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader, val_loader