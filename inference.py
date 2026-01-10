import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from app.config import Config
from app.models.classifier import get_model, get_poison_detector
from app.models.detector import PoisonDetector
import os

class InferenceSystem:
    """
    Système d'inférence pour détecter les attaques et classifier les images
    """
    
    def __init__(self, model_path="global_model_final.pth", detector_path="poison_detector.pth"):
        # Charger le modèle de classification
        self.model = get_model(pretrained=False)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            print(f"✓ Modèle chargé depuis {model_path}")
        else:
            print(f"⚠️ Fichier {model_path} non trouvé, utilisation du modèle pré-entraîné")
        
        self.model.eval()
        
        # Charger le détecteur d'attaques
        self.poison_detector = PoisonDetector(self.model)
        if os.path.exists(detector_path):
            self.poison_detector.load_detector(detector_path)
            print(f"✓ Détecteur chargé depuis {detector_path}")
        else:
            print(f"⚠️ Fichier {detector_path} non trouvé")
        
        # Transformation pour les images
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ['NORMAL', 'PNEUMONIA']
    
    def predict_single_image(self, image_path, check_adversarial=True):
        """
        Prédire la classe d'une seule image et détecter les attaques
        
        Args:
            image_path: Chemin vers l'image
            check_adversarial: Vérifier si l'image est adversariale
        
        Returns:
            dict avec la prédiction et les détections
        """
        # Charger et prétraiter l'image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
        
        results = {
            'image_path': image_path,
            'is_adversarial': False,
            'adversarial_confidence': 0.0,
            'prediction': None,
            'confidence': 0.0,
            'all_probabilities': {}
        }
        
        # Vérifier si l'image est adversariale
        if check_adversarial:
            is_poisoned, confidence = self.poison_detector.detect_poison(image_tensor)
            
            # Correction : utiliser .item() directement pour scalaires 0-dim
            results['is_adversarial'] = bool(is_poisoned.item())
            results['adversarial_confidence'] = float(confidence.item())
        
        # Faire la prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        results['prediction'] = self.classes[predicted_class]
        results['confidence'] = confidence
        results['all_probabilities'] = {
            self.classes[i]: float(probabilities[i].item()) 
            for i in range(len(self.classes))
        }
        
        return results
    
    def visualize_prediction(self, image_path, results):
        """
        Visualiser l'image avec la prédiction et la détection d'attaque
        """
        # Charger l'image
        image = Image.open(image_path)
        
        # Créer la figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Titre avec la prédiction
        title = f"Prédiction: {results['prediction']} ({results['confidence']*100:.2f}%)\n"
        
        # Ajouter l'information sur l'attaque
        if results['is_adversarial']:
            title += f"⚠️ ATTAQUE DÉTECTÉE (confiance: {results['adversarial_confidence']*100:.2f}%)"
            title_color = 'red'
        else:
            title += f"✓ Image propre (confiance: {results['adversarial_confidence']*100:.2f}%)"
            title_color = 'green'
        
        ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
        
        # Afficher les probabilités
        prob_text = "Probabilités:\n"
        for class_name, prob in results['all_probabilities'].items():
            prob_text += f"  {class_name}: {prob*100:.2f}%\n"
        
        plt.figtext(0.15, 0.02, prob_text, fontsize=10, ha='left')
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, image_paths):
        """
        Prédire sur un lot d'images
        """
        results = []
        
        print(f"\n🔍 Analyse de {len(image_paths)} images...\n")
        
        for i, image_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}")
            
            result = self.predict_single_image(image_path)
            results.append(result)
            
            # Afficher le résultat
            status = "⚠️ ATTAQUE" if result['is_adversarial'] else "✓ PROPRE"
            print(f"  {status} | Prédiction: {result['prediction']} ({result['confidence']*100:.2f}%)")
            print()
        
        return results

def demo():
    """
    Démonstration du système d'inférence
    """
    print("="*70)
    print("  🔍 SYSTÈME DE DÉTECTION ET CLASSIFICATION")
    print("="*70)
    
    # Créer le système d'inférence
    inference = InferenceSystem()
    
    # Exemple d'utilisation
    test_images_dir = os.path.join(Config.DATASET_PATH, "test", "NORMAL")
    
    if os.path.exists(test_images_dir):
        # Prendre quelques images de test
        image_files = [
            os.path.join(test_images_dir, f) 
            for f in os.listdir(test_images_dir)[:5]
            if f.endswith(('.jpeg', '.jpg', '.png'))
        ]
        
        # Prédire sur ces images
        results = inference.batch_predict(image_files)
        
        # Afficher les statistiques
        num_adversarial = sum(1 for r in results if r['is_adversarial'])
        print("\n" + "="*70)
        print("RÉSUMÉ:")
        print(f"  Images analysées: {len(results)}")
        print(f"  Attaques détectées: {num_adversarial}")
        print(f"  Images propres: {len(results) - num_adversarial}")
        print("="*70)
    else:
        print(f"\n⚠️ Répertoire {test_images_dir} non trouvé")
        print("Utilisez cette classe dans votre code:")
        print("\n  inference = InferenceSystem()")
        print("  result = inference.predict_single_image('path/to/image.jpg')") 
        print("  inference.visualize_prediction('path/to/image.jpg', result)")

if __name__ == "__main__":
    demo()
