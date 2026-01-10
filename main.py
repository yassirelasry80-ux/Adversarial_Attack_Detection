import torch
import numpy as np
import random
import os
from app.config import Config
from app.data.loader import create_federated_datasets, get_dataloaders
from app.models.classifier import get_model
from app.attacks.adversarial import AdversarialAttacks
import argparse
from app.models.detector import PoisonDetector, AutoEncoderDetector
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from app.federated.learning import FederatedLearning

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
    parser = argparse.ArgumentParser(description='Adversarial Attack Detection')
    parser.add_argument('--method', type=str, default='supervised', 
                      choices=['supervised', 'autoencoder'],
                      help='Method for detection: supervised or autoencoder')
    args = parser.parse_args()

    # Configuration initiale
    set_seed()
    
    print_header(f"🚀 SYSTÈME DE DÉTECTION D'ATTAQUES ADVERSARIALES ({args.method.upper()})")
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
    
    # Étape 4: Entraînement et Comparaison des détecteurs
    print_header("🔍 ÉTAPE 4: ENTRAÎNEMENT ET COMPARAISON DES DÉTECTEURS")
    
    # --- 1. Entraînement Supervisé ---
    print("\n[1/2] Entraînement du Détecteur Supervisé (MLP)...")
    detector_sup = PoisonDetector(pretrained_model)
    detector_sup.train_detector(train_set, val_data=val_set, epochs=5) # Reduced epochs for speed
    detector_sup.save_detector("poison_detector.pth")
    
    # --- 2. Entraînement Auto-Encodeur ---
    print("\n[2/2] Entraînement du Détecteur Auto-Encodeur...")
    detector_ae = AutoEncoderDetector()
    detector_ae.train_detector(train_set, val_data=val_set, epochs=5) # Reduced epochs for speed
    detector_ae.save_detector("autoencoder.pth")
    
    # --- 3. Comparaison ---
    print_header("📊 COMPARAISON DES PERFORMANCES")
    
    # Evaluation sur le set de validation (mixte)
    print("Évaluation sur le dataset de validation mixte (Clean + Attacks)...")
    
    def evaluate_detector(det_model, val_data, name):
        correct = 0
        total = 0
        
        # Préparer les données
        images = torch.cat([item[0] for item in val_data]).to(Config.DEVICE)
        labels = torch.tensor([item[2] for item in val_data]).to(Config.DEVICE) # item[2] is is_adversarial
        
        is_poisoned, _ = det_model.detect_poison(images)
        is_poisoned = is_poisoned.to(Config.DEVICE)
        
        correct = (is_poisoned == labels).sum().item()
        total = len(labels)
        acc = 100. * correct / total
        return acc

    acc_sup = evaluate_detector(detector_sup, val_set, "Supervisé")
    acc_ae = evaluate_detector(detector_ae, val_set, "Auto-Encodeur")
    
    print(f"\nPrécision de détection (Accuracy):")
    print(f"  1. Supervisé (MLP)       : {acc_sup:.2f}%")
    print(f"  2. Auto-Encodeur (Seuil) : {acc_ae:.2f}%")
    
    # --- 4. Choix de l'utilisateur ---
    print("\n" + "="*50)
    print("🤔 CHOIX DU DÉTECTEUR POUR LA FÉDÉRATION")
    print("="*50)
    print("Quel détecteur voulez-vous utiliser pour protéger les hôpitaux ?")
    print("1: Supervisé (MLP)")
    print("2: Auto-Encodeur (Non-supervisé)")
    
    while True:
        choice = input("\nVotre choix (1 ou 2): ").strip()
        if choice == "1":
            selected_detector = detector_sup
            print(">> Vous avez choisi: SUPERVISÉ")
            break
        elif choice == "2":
            selected_detector = detector_ae
            print(">> Vous avez choisi: AUTO-ENCODEUR")
            break
        else:
            print("Choix invalide, réessayez.")

    # Étape 5 & 6: Déploiement du Détecteur et Apprentissage Fédéré
    print_header("🏥 ÉTAPES 5 & 6: DÉPLOIEMENT ET FÉDÉRATION")
    print(f"Déploiement du détecteur {choice} (Sélectionné) à l'entrée de chaque hôpital...")
    
    # Préparer les sources de données pour chaque hôpital
    # H1, H2, H4 sont propres (Simulation normale)
    # H3 est attaqué (Simulation d'attaque)
    
    # Pour H3, on doit générer l'attaque MAINTENANT si ce n'est pas fait
    print("\n[Simulation] Génération de l'attaque sur l'Hôpital 3...")
    h3_loader = DataLoader(hospital_datasets[2], batch_size=Config.BATCH_SIZE)
    h3_attacked = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model, h3_loader, attack_type='pgd', ratio=0.5
    )
    # Convertir H3 en TensorDataset pour faciliter la suite
    h3_imgs = torch.cat([item[0] for item in h3_attacked])
    h3_lbls = torch.tensor([item[1].item() for item in h3_attacked])
    # Note: item[2] est le flag is_adv, on ne l'utilise pas pour le filtrage (c'est le détecteur qui devine)
    dataset_h3_attacked = TensorDataset(h3_imgs, h3_lbls)
    
    # Liste des datasets "bruts" qui arrivent à chaque hôpital
    raw_datasets_per_hospital = [
        hospital_datasets[0],      # H1 (Clean)
        hospital_datasets[1],      # H2 (Clean)
        dataset_h3_attacked,       # H3 (Attaqué!)
        hospital_datasets[3]       # H4 (Clean)
    ]
    
    fl_ready_datasets = []
    
    for i, raw_ds in enumerate(raw_datasets_per_hospital):
        print(f"\n🔒 Filtrage Hôpital {i+1}...")
        
        # Créer loader temporaire
        loader = DataLoader(raw_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        # Le détecteur filtre (rejette ce qu'il pense être des attaques)
        clean_data_list = selected_detector.filter_clean_data(loader)
        
        # Reconvertir en Dataset pytorch
        if len(clean_data_list) > 0:
            c_imgs = torch.stack([item[0] for item in clean_data_list])
            c_lbls = torch.stack([item[1] for item in clean_data_list])
            clean_ds = TensorDataset(c_imgs, c_lbls)
            fl_ready_datasets.append(clean_ds)
            print(f"  -> Données acceptées pour FL: {len(clean_ds)}/{len(raw_ds)}")
        else:
            print(f"  -> ⚠️ TOUTES les données ont été rejetées par le détecteur !")
    
    # Lancement du FL
    print_header("🚀 LANCEMENT DE L'APPRENTISSAGE FÉDÉRÉ")
    
    if len(fl_ready_datasets) == 0:
        print("❌ Erreur: Plus aucune donnée disponible après filtrage.")
        return

    # Créer un nouveau modèle global
    global_model = get_model(pretrained=True)
    
    # Initialiser l'apprentissage fédéré
    fed_learning = FederatedLearning(global_model)
    
    # Entraîner de manière fédérée AVEC les données filtrées
    final_model = fed_learning.federated_training(fl_ready_datasets, test_loader=test_loader)
    
    # Étape 7: Évaluation finale
    print_header("📊 ÉTAPE 7: ÉVALUATION FINALE DU MODÈLE GLOBAL")
    
    # Évaluer le modèle global
    accuracy = fed_learning.evaluate_global_model(test_loader)
    
    # Sauvegarder les modèles
    print_header("💾 SAUVEGARDE DES RESULTATS")
    
    if choice == "1":
        global_model_name = "global_model_supervised.pth"
    else:
        global_model_name = "global_model_autoencoder.pth"
        
    fed_learning.save_global_model(global_model_name)
    
    print_header("✅ PROCESSUS COMPLET TERMINÉ")
    print("Résumé:")
    print("1. Détecteurs générés et comparés.")
    print("2. Détecteur choisi déployé sur TOUS les hôpitaux.")
    print("3. Hôpital 3 (Attaqué) a été filtré.")
    print("4. Hôpitaux 1, 2, 4 (Sains) ont été vérifiés.")
    print(f"5. Apprentissage Fédéré exécuté sur les données validées.")
    print(f"6. Modèle global sauvegardé sous: {global_model_name}")

if __name__ == "__main__":
    main()