import torch
import torch.nn as nn
import torchvision.models as models
from app.config import Config

class ChestXRayModel(nn.Module):
    """
    Modèle basé sur ResNet18 pour la classification de radiographies thoraciques
    Optimisé pour RTX 4060 8GB
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(ChestXRayModel, self).__init__()
        
        # Utiliser ResNet18 (plus léger que ResNet50)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remplacer la dernière couche pour notre tâche
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extraire les features avant la classification"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class PoisonDetectionModel(nn.Module):
    """
    Modèle de détection d'attaques adversariales
    Analyse les caractéristiques pour détecter les données empoisonnées
    """
    def __init__(self, input_dim=512):
        super(PoisonDetectionModel, self).__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.detector(x)

def get_model(pretrained=True):
    """Créer et retourner le modèle"""
    model = ChestXRayModel(pretrained=pretrained)
    return model.to(Config.DEVICE)

def get_poison_detector(input_dim=512):
    """Créer et retourner le détecteur de poison"""
    detector = PoisonDetectionModel(input_dim=input_dim)
    return detector.to(Config.DEVICE)

class AutoEncoder(nn.Module):
    """
    AutoEncodeur Convolutionnel pour la détection d'anomalies (attaques)
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56 -> 28
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112 -> 224
            nn.Sigmoid() # Pour avoir des valeurs entre 0 et 1 comme les images normalisées
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_autoencoder():
    """Créer et retourner l'AutoEncodeur"""
    model = AutoEncoder()
    return model.to(Config.DEVICE)