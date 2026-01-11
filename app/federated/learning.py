import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from app.config import Config

class FederatedLearning:
    """
    Implémentation de l'apprentissage fédéré
    """
    
    def __init__(self, global_model):
        self.global_model = global_model
        self.local_models = []
    
    def train_local_model(self, model, dataloader, epochs=5):
        """
        Entraîner un modèle local sur les données d'un hôpital
        
        Args:
            model: Modèle local
            dataloader: DataLoader pour cet hôpital
            epochs: Nombre d'époques
        
        Returns:
            Modèle entraîné
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in dataloader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistiques
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                # Mise à jour de la barre de progression (tqdm) serait top, mais on va juste print à la fin
            
            accuracy = 100. * correct / total
            print(f"      Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {accuracy:.2f}%")
        
        return model
    
    def aggregate_models(self, local_models):
        """
        Agrégation FedAvg: moyenne des poids des modèles locaux
        
        Args:
            local_models: Liste des modèles locaux entraînés
        """
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            # Calculer la moyenne des poids
            global_dict[key] = torch.stack([
                local_models[i].state_dict()[key].float() 
                for i in range(len(local_models))
            ]).mean(0)
        
        self.global_model.load_state_dict(global_dict)
    
    def federated_training(self, hospital_datasets, test_loader=None, num_rounds=Config.FEDERATED_ROUNDS):
        """
        Entraînement fédéré complet
        
        Args:
            hospital_datasets: Liste des datasets pour chaque hôpital
            test_loader: DataLoader pour évaluer le modèle global à chaque round
            num_rounds: Nombre de rounds fédérés
        """
        print(f"\n🏥 Démarrage de l'apprentissage fédéré avec {len(hospital_datasets)} hôpitaux")
        print(f"   Nombre de rounds: {num_rounds}\n")
        
        for round_num in range(num_rounds):
            print(f"\n{'='*60}")
            print(f"ROUND FÉDÉRÉ {round_num + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            local_models = []
            
            # Entraîner chaque modèle local
            for hospital_id, dataset in enumerate(hospital_datasets):
                print(f"\n🏥 Hôpital {hospital_id + 1}/{len(hospital_datasets)}")
                print(f"   Taille du dataset: {len(dataset)} images")
                
                # Créer un DataLoader pour cet hôpital
                dataloader = DataLoader(
                    dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
                
                # Copier le modèle global
                local_model = copy.deepcopy(self.global_model)
                
                # Entraîner localement
                print("   Entraînement local...")
                local_model = self.train_local_model(
                    local_model, 
                    dataloader, 
                    epochs=Config.LOCAL_EPOCHS
                )
                
                local_models.append(local_model)
                print("   ✓ Entraînement local terminé")
            
            # Agréger les modèles
            print(f"\n🔄 Agrégation des {len(local_models)} modèles locaux...")
            self.aggregate_models(local_models)
            print("✓ Agrégation terminée")
            
            # Évaluer le modèle global (et logging)
            if test_loader:
                print(f"\n📊 Évaluation Round {round_num + 1}:")
                self.evaluate_global_model(test_loader)
            
            print(f"\n✓ Round {round_num + 1} terminé")
        
        print(f"\n{'='*60}")
        print("✓ APPRENTISSAGE FÉDÉRÉ TERMINÉ")
        print(f"{'='*60}\n")
        
        return self.global_model
    
    def evaluate_global_model(self, test_loader):
        """
        Évaluer le modèle global sur le dataset de test
        
        Args:
            test_loader: DataLoader de test
        
        Returns:
            Accuracy du modèle
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        print("\n📊 Évaluation du modèle global...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Évaluation"):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                outputs = self.global_model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"\n✓ Accuracy du modèle global: {accuracy:.2f}%")
        
        return accuracy
    
    def save_global_model(self, path="global_model.pth"):
        """Sauvegarder le modèle global"""
        torch.save(self.global_model.state_dict(), path)
        print(f"✓ Modèle global sauvegardé dans {path}")
    
    def load_global_model(self, path="global_model.pth"):
        """Charger un modèle global"""
        self.global_model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        print(f"✓ Modèle global chargé depuis {path}")