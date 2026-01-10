import torch
import numpy as np
import random
import os
from app.config import Config
from app.data.loader import create_federated_datasets, get_dataloaders
from app.models.classifier import get_model
from app.attacks.adversarial import AdversarialAttacks
from app.models.detector import PoisonDetector
from app.federated.learning import FederatedLearning
from torch.utils.data import DataLoader, ConcatDataset

def set_seed(seed=Config.RANDOM_SEED):
    """Fixer les seeds pour la reproductibilité"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_header(text):
    """Afficher un en-tête formaté"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def main():
    # Configuration initiale
    set_seed()
    
    print_header("🚀 SYSTÈME DE DÉTECTION D'ATTAQUES ADVERSARIALES")
    print(f"Device utilisé: {Config.DEVICE}")
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Étape 1: Charger les données
    print_header("📁 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    
    if not os.path.exists(Config.DATASET_PATH):
        print("❌ Dataset non trouvé!")
        print("Exécutez d'abord: python download_data.py")
        return
    
    # Créer les datasets fédérés
    hospital_datasets = create_federated_datasets(Config.DATASET_PATH)
    print(f"✓ {Config.NUM_HOSPITALS} hôpitaux créés")
    for i, dataset in enumerate(hospital_datasets):
        print(f"  - Hôpital {i+1}: {len(dataset)} images")
    
    # Charger les données de test
    test_loader, val_loader = get_dataloaders(Config.DATASET_PATH)
    print(f"✓ Dataset de test: {len(test_loader.dataset)} images")
    print(f"✓ Dataset de validation: {len(val_loader.dataset)} images")
    
    # Étape 2: Pré-entraînement du modèle
    print_header("🧠 ÉTAPE 2: PRÉ-ENTRAÎNEMENT DU MODÈLE")
    
    pretrained_model = get_model(pretrained=True)
    print("✓ Modèle ResNet18 pré-entraîné chargé")
    
    # Étape 3: Génération d'attaques adversariales
    print_header("⚔️ ÉTAPE 3: GÉNÉRATION D'ATTAQUES ADVERSARIALES")
    
    print("Génération d'exemples adversariaux FGSM et PGD...")
    
    # Utiliser les deux premiers datasets d'hôpitaux pour l'entraînement du détecteur
    print("Assemblage des données des Hôpitaux 1 et 2 pour l'entraînement du détecteur...")
    detector_training_data = ConcatDataset([hospital_datasets[0], hospital_datasets[1]])
    
    sample_loader = DataLoader(
        detector_training_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Générer des exemples adversariaux FGSM
    print("\n🎯 Génération d'attaques FGSM...")
    fgsm_data = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model, 
        sample_loader, 
        attack_type='fgsm',
        ratio=0.3
    )
    print(f"✓ {len(fgsm_data)} exemples générés (30% adversariaux)")
    
    # Générer des exemples adversariaux PGD
    print("\n🎯 Génération d'attaques PGD...")
    pgd_data = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model,
        sample_loader,
        attack_type='pgd',
        ratio=0.3
    )
    print(f"✓ {len(pgd_data)} exemples générés (30% adversariaux)")
    
    # Combiner les données
    adversarial_train_data = fgsm_data + pgd_data
    print(f"\n✓ Total d'exemples d'entraînement: {len(adversarial_train_data)}")
    
    # Split Train/Val (80/20)
    random.shuffle(adversarial_train_data)
    split_idx = int(0.8 * len(adversarial_train_data))
    train_set = adversarial_train_data[:split_idx]
    val_set = adversarial_train_data[split_idx:]
    print(f"✓ Split Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Étape 4: Entraînement du détecteur d'attaques
    print_header("🔍 ÉTAPE 4: ENTRAÎNEMENT DU DÉTECTEUR")
    
    poison_detector = PoisonDetector(pretrained_model)
    poison_detector.train_detector(train_set, val_data=val_set, epochs=10)
    poison_detector.save_detector("poison_detector.pth")
    
    # Étape 5: Filtrage des données empoisonnées
    print_header("🧹 ÉTAPE 5: FILTRAGE DES DONNÉES")
    
    # Créer un loader avec des données potentiellement empoisonnées
    # On utilise l'Hôpital 3 car les 1 et 2 ont servi à l'entraînement
    poisoned_loader = DataLoader(
        hospital_datasets[2],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Générer des attaques sur ce dataset
    print("Contamination du dataset avec des attaques...")
    attacked_data = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model,
        poisoned_loader,
        attack_type='pgd',
        ratio=0.4
    )
    
    # Créer un loader avec les données attaquées
    from torch.utils.data import TensorDataset
    attacked_images = torch.cat([item[0] for item in attacked_data])
    attacked_labels = torch.tensor([item[1].item() for item in attacked_data])
    attacked_dataset = TensorDataset(attacked_images, attacked_labels)
    attacked_loader = DataLoader(
        attacked_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Filtrer les données
    # Charger le meilleur détecteur pour le filtrage
    if os.path.exists("poison_detector_best.pth"):
        poison_detector.load_detector("poison_detector_best.pth")
        print("✓ Meilleur détecteur chargé pour le filtrage")
        
    clean_data = poison_detector.filter_clean_data(attacked_loader)
    
    # Étape 6: Apprentissage fédéré avec données propres
    print_header("🏥 ÉTAPE 6: APPRENTISSAGE FÉDÉRÉ")
    
    # Créer un nouveau modèle global
    global_model = get_model(pretrained=True)
    
    # Initialiser l'apprentissage fédéré
    fed_learning = FederatedLearning(global_model)
    
    # Entraîner de manière fédérée
    final_model = fed_learning.federated_training(hospital_datasets, test_loader=test_loader)
    
    # Étape 7: Évaluation finale
    print_header("📊 ÉTAPE 7: ÉVALUATION FINALE")
    
    # Évaluer le modèle global
    accuracy = fed_learning.evaluate_global_model(test_loader)
    
    # Tester la robustesse contre les attaques
    print("\n🛡️ Test de robustesse contre les attaques...")
    
    # Générer des exemples adversariaux sur le test set
    test_adv_fgsm = []
    test_adv_pgd = []
    
    for images, labels in test_loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # FGSM
        adv_fgsm = AdversarialAttacks.fgsm_attack(final_model, images, labels)
        test_adv_fgsm.append((adv_fgsm, labels))
        
        # PGD
        adv_pgd = AdversarialAttacks.pgd_attack(final_model, images, labels)
        test_adv_pgd.append((adv_pgd, labels))
    
    # Évaluer sur les données adversariales
    print("\nÉvaluation sur données originales:")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Sauvegarder les modèles
    print_header("💾 SAUVEGARDE DES MODÈLES")
    fed_learning.save_global_model("global_model_final.pth")
    
    print_header("✅ PROCESSUS TERMINÉ AVEC SUCCÈS")
    print("Fichiers générés:")
    print("  - poison_detector.pth")
    print("  - global_model_final.pth")
    print("\nVous pouvez maintenant utiliser ces modèles pour:")
    print("  1. Détecter les attaques adversariales")
    print("  2. Classifier les radiographies thoraciques")
    print("  3. Poursuivre l'entraînement fédéré")

if __name__ == "__main__":
    main()