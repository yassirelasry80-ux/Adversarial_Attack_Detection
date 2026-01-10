import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from app.config import Config
from app.models.classifier import get_poison_detector, get_autoencoder

class PoisonDetector:
    """
    Détecteur d'attaques adversariales utilisant le Deep Learning
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.detector = get_poison_detector()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
    
    def extract_features(self, images):
        """Extraire les features des images"""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor.get_features(images)
        return features
    
    def train_detector(self, train_data, val_data=None, epochs=10):
        """
        Entraîner le détecteur de poison
        
        Args:
            train_data: Liste de tuples (image, label, is_adversarial)
            val_data: Liste de tuples pour la validation
            epochs: Nombre d'époques
        """
        print("\n🔍 Entraînement du détecteur d'attaques adversariales...")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.detector.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Mélanger les données
            import random
            random.shuffle(train_data)
            
            # Créer des mini-batches
            batch_size = Config.BATCH_SIZE
            num_batches = len(train_data) // batch_size
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch_idx in progress_bar:
                # Préparer le batch
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                batch = train_data[batch_start:batch_end]
                
                images = torch.cat([item[0] for item in batch]).to(Config.DEVICE)
                is_adv = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(Config.DEVICE)
                
                # Extraire les features
                features = self.extract_features(images)
                
                # Forward pass
                predictions = self.detector(features).squeeze()
                loss = self.criterion(predictions, is_adv)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistiques
                total_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct += (predicted == is_adv).sum().item()
                total += len(is_adv)
                
                # Mise à jour de la barre de progression
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            train_loss = total_loss / num_batches
            train_acc = 100. * correct / total
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%")
            
            # Validation
            if val_data:
                self.detector.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0
                
                # Batcher pour la validation sinon OOM
                val_num_batches = len(val_data) // batch_size
                
                with torch.no_grad():
                    for i in range(val_num_batches + 1):
                        start = i * batch_size
                        end = min(start + batch_size, len(val_data))
                        if start >= end: break
                        
                        batch = val_data[start:end]
                        images = torch.cat([item[0] for item in batch]).to(Config.DEVICE)
                        is_adv = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(Config.DEVICE)
                        
                        features = self.extract_features(images)
                        preds = self.detector(features).squeeze()
                        loss = self.criterion(preds, is_adv)
                        
                        val_loss += loss.item() * len(batch)
                        predicted = (preds > 0.5).float()
                        val_correct += (predicted == is_adv).sum().item()
                        val_total += len(batch)
                
                if val_total > 0:
                    val_acc = 100. * val_correct / val_total
                    val_avg_loss = val_loss / val_total
                    print(f"Epoch {epoch+1}: Val Loss = {val_avg_loss:.4f}, Val Acc = {val_acc:.2f}%")
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        self.save_detector("poison_detector_best.pth")
                        print(f"✓ Nouveau meilleur modèle sauvegardé (Acc: {best_acc:.2f}%)")    
    def detect_poison(self, images, threshold=Config.DETECTION_THRESHOLD):
        """
        Détecter si des images sont adversariales
        
        Args:
            images: Images à tester
            threshold: Seuil de décision
        
        Returns:
            Tensor de booléens indiquant si chaque image est adversariale
        """
        self.detector.eval()
        
        with torch.no_grad():
            features = self.extract_features(images)
            predictions = self.detector(features).squeeze()
            
            # Utiliser un seuil pour la décision
            is_poisoned = predictions > threshold
        
        return is_poisoned, predictions
    
    def filter_clean_data(self, dataloader, output_path=Config.CLEAN_DATA_PATH):
        """
        Filtrer les données propres en retirant les exemples adversariaux
        
        Args:
            dataloader: DataLoader contenant les données à filtrer
            output_path: Chemin pour sauvegarder les données propres
        
        Returns:
            Liste de données propres
        """
        print("\n🧹 Filtrage des données empoisonnées...")
        
        clean_data = []
        total_images = 0
        poisoned_images = 0
        
        self.detector.eval()
        
        for images, labels in tqdm(dataloader, desc="Filtrage"):
            images = images.to(Config.DEVICE)
            
            # Détecter les images empoisonnées
            is_poisoned, confidence = self.detect_poison(images)
            
            # Garder seulement les images propres
            for i in range(len(images)):
                total_images += 1
                if not is_poisoned[i]:
                    clean_data.append((images[i].cpu(), labels[i].cpu()))
                else:
                    poisoned_images += 1
        
        detection_rate = 100. * poisoned_images / total_images
        print(f"\n✓ Filtrage terminé:")
        print(f"  - Total d'images: {total_images}")
        print(f"  - Images empoisonnées détectées: {poisoned_images} ({detection_rate:.2f}%)")
        print(f"  - Images propres conservées: {len(clean_data)}")
        
        return clean_data
    
    def save_detector(self, path="poison_detector.pth"):
        """Sauvegarder le détecteur"""
        torch.save({
            'detector_state_dict': self.detector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✓ Détecteur sauvegardé dans {path}")
    
    def load_detector(self, path="poison_detector.pth"):
        """Charger un détecteur pré-entraîné"""
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.detector.load_state_dict(checkpoint['detector_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Détecteur chargé depuis {path}")

class AutoEncoderDetector:
    """
    Détecteur d'attaques basé sur un AutoEncodeur (Approche Non-Supervisée)
    Principe: L'AutoEncodeur apprend à reconstruire les images propres.
    Les images adversariales auront une erreur de reconstruction (MSE) plus élevée.
    """
    
    def __init__(self):
        self.detector = get_autoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
        # Seuil par défaut, sera calibré après l'entraînement
        self.threshold = 0.05 
        
    def train_detector(self, train_data, val_data=None, epochs=10):
        """
        Entraîner l'AutoEncodeur UNIQUEMENT sur les données propres
        
        Args:
            train_data: Liste de tuples (image, label, is_adversarial)
            val_data: Liste de tuples pour la validation
            epochs: Nombre d'époques
        """
        print("\n🔍 Entraînement de l'AutoEncodeur (Détection d'anomalies)...")
        
        # Filtrer pour ne garder que les images propres (is_adversarial == 0)
        # Contrairement au classifieur supervisé qui a besoin des deux classes
        clean_train_data = [item for item in train_data if item[2] == 0]
        
        if len(clean_train_data) == 0:
            print("⚠️ AVERTISSEMENT: Aucune donnée propre trouvée pour l'entraînement!")
            return
            
        print(f"ℹ️ Entraînement sur {len(clean_train_data)} images propres uniquement")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.detector.train()
            total_loss = 0
            num_batches = 0
            
            # Mélanger
            import random
            random.shuffle(clean_train_data)
            
            # Batcher manuellement comme avant
            batch_size = Config.BATCH_SIZE
            num_batches_total = len(clean_train_data) // batch_size
            
            progress_bar = tqdm(range(num_batches_total), desc=f"Epoch {epoch+1}/{epochs} [AE Train]")
            
            for batch_idx in progress_bar:
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                batch = clean_train_data[batch_start:batch_end]
                
                images = torch.cat([item[0] for item in batch]).to(Config.DEVICE)
                
                # Forward pass - Reconstruction
                reconstructed = self.detector(images)
                loss = self.criterion(reconstructed, images)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'mse_loss': f'{loss.item():.6f}'})
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}: Train MSE Loss = {avg_loss:.6f}")
            
            # Validation (aussi sur données propres pour vérifier la convergence)
            if val_data:
                self.detector.eval()
                val_loss = 0
                val_batches = 0
                
                clean_val_data = [item for item in val_data if item[2] == 0]
                val_num_iters = len(clean_val_data) // batch_size
                
                with torch.no_grad():
                    for i in range(val_num_iters):
                        start = i * batch_size
                        end = start + batch_size
                        batch = clean_val_data[start:end]
                        images = torch.cat([item[0] for item in batch]).to(Config.DEVICE)
                        
                        recon = self.detector(images)
                        loss = self.criterion(recon, images)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / max(1, val_batches)
                print(f"Epoch {epoch+1}: Val MSE Loss = {avg_val_loss:.6f}")
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    self.save_detector("autoencoder_best.pth")
                    print(f"✓ Meilleur modèle sauvegardé (Loss: {best_loss:.6f})")
                    
        # Calibrage du seuil à la fin de l'entraînement
        # 1. Charger les meilleurs poids pour la calibration
        self.load_detector("autoencoder_best.pth")
        
        # 2. Calibrer le seuil avec ces meilleurs poids
        self.calibrate_threshold(clean_train_data)
        
        # 3. Resauvegarder le meilleur modèle AVEC le bon seuil
        self.save_detector("autoencoder_best.pth")
        print("✓ Seuil calibré sauvegardé dans autoencoder_best.pth")

    def calibrate_threshold(self, clean_data):
        """Définir le seuil basé sur l'erreur max de reconstruction des données propres + marge"""
        self.detector.eval()
        print("\n📏 Calibrage du seuil de détection...")
        
        errors = []
        batch_size = Config.BATCH_SIZE
        num_batches = len(clean_data) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Calcul erreurs"):
                batch = clean_data[i*batch_size:(i+1)*batch_size]
                images = torch.cat([item[0] for item in batch]).to(Config.DEVICE)
                recon = self.detector(images)
                
                # Erreur par image (pas moyenne du batch)
                loss_per_image = torch.mean((images - recon)**2, dim=[1, 2, 3])
                errors.extend(loss_per_image.cpu().numpy())
        
        import numpy as np
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        # Seuil = Moyenne + 3 * Ecart-type (couvre 99.7% des données normales si gaussien)
        self.threshold = mean_error + 3 * std_error
        
        print(f"✓ Seuil calibré: {self.threshold:.6f} (Moyenne: {mean_error:.6f}, Std: {std_error:.6f})")
        
    def detect_poison(self, images, threshold=None):
        """
        Détecter si des images sont mauvaises (erreur de reconstruction élevée)
        """
        if threshold is None:
            threshold = self.threshold
            
        self.detector.eval()
        with torch.no_grad():
            reconstructed = self.detector(images)
            # Erreur quadratique moyenne par image
            mse_per_image = torch.mean((images - reconstructed)**2, dim=[1, 2, 3])
            
            is_poisoned = mse_per_image > threshold
            
        return is_poisoned, mse_per_image

    def filter_clean_data(self, dataloader, output_path=Config.CLEAN_DATA_PATH):
        # Utilise la même logique que PoisonDetector mais appelle son propre detect_poison
        print("\n🧹 Filtrage avec AutoEncodeur...")
        clean_data = []
        total = 0
        poisoned = 0
        
        self.detector.eval()
        for images, labels in tqdm(dataloader, desc="Filtrage AE"):
            images = images.to(Config.DEVICE)
            is_poisoned, _ = self.detect_poison(images)
            
            for i in range(len(images)):
                total += 1
                if not is_poisoned[i]:
                    clean_data.append((images[i].cpu(), labels[i].cpu()))
                else:
                    poisoned += 1
                    
        print(f"✓ (AE) Filtrage terminé: {poisoned}/{total} images rejetées")
        return clean_data

    def save_detector(self, path="autoencoder.pth"):
        torch.save({
            'detector_state_dict': self.detector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold
        }, path)
        print(f"✓ AutoEncodeur sauvegardé dans {path}")

    def load_detector(self, path="autoencoder.pth"):
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        self.detector.load_state_dict(checkpoint['detector_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint.get('threshold', 0.05)
        print(f"✓ AutoEncodeur chargé depuis {path} (Seuil: {self.threshold:.6f})")
