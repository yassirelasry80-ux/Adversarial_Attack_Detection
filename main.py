import torch
import numpy as np
import random
import os
import argparse
from app.config import Config
from app.data.loader import create_federated_datasets, get_dataloaders
from app.models.classifier import get_model
from app.attacks.adversarial import AdversarialAttacks
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

def train_detectors_step():
    print_header("🚀 PARTIE 1: ENTRAÎNEMENT DES DÉTECTEURS")
    
    # 1. Charger modèle pré-entraîné (pour générer attaques)
    pretrained_model = get_model(pretrained=True)
    
    # 2. Préparer données (H1 + H2)
    hospital_datasets = create_federated_datasets(Config.DATASET_PATH)
    detector_training_data = ConcatDataset([hospital_datasets[0], hospital_datasets[1]])
    
    sample_loader = DataLoader(
        detector_training_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # 3. Générer Attaques (FGSM + PGD)
    print("\n⚔️ Génération d'attaques pour l'entraînement...")
    fgsm_data = AdversarialAttacks.generate_adversarial_dataset(pretrained_model, sample_loader, attack_type='fgsm', ratio=0.3)
    pgd_data = AdversarialAttacks.generate_adversarial_dataset(pretrained_model, sample_loader, attack_type='pgd', ratio=0.3)
    
    adversarial_train_data = fgsm_data + pgd_data
    random.shuffle(adversarial_train_data)
    
    # Split Train/Val
    split_idx = int(0.8 * len(adversarial_train_data))
    train_set = adversarial_train_data[:split_idx]
    val_set = adversarial_train_data[split_idx:]
    
    # 4. Entraîner Supervised Detector
    print_header("🧠 ENTRAÎNEMENT DÉTECTEUR SUPERVISÉ (MLP)")
    detector_sup = PoisonDetector(pretrained_model)
    detector_sup.train_detector(train_set, val_data=val_set, epochs=Config.DETECTOR_EPOCHS)
    detector_sup.save_detector("poison_detector.pth")
    
    # 5. Entraîner AutoEncoder Detector
    print_header("🧠 ENTRAÎNEMENT DÉTECTEUR AUTO-ENCODEUR")
    detector_ae = AutoEncoderDetector()
    detector_ae.train_detector(train_set, val_data=val_set, epochs=Config.DETECTOR_EPOCHS)
    detector_ae.save_detector("autoencoder.pth")
    
    print("\n✅ Entraînement terminé ! Les détecteurs sont sauvegardés.")

def federated_learning_step(method):
    print_header(f"🚀 PARTIE 2: APPRENTISSAGE FÉDÉRÉ ({method.upper()})")
    
    # 1. Charger modèle global & Data
    pretrained_model = get_model(pretrained=True)
    hospital_datasets = create_federated_datasets(Config.DATASET_PATH)
    test_loader, _ = get_dataloaders(Config.DATASET_PATH)
    
    # 2. Charger le bon détecteur
    if method == 'supervised':
        detector = PoisonDetector(pretrained_model)
        if os.path.exists("poison_detector_best.pth"):
             detector.load_detector("poison_detector_best.pth")
        else:
             detector.load_detector("poison_detector.pth")
    else:
        detector = AutoEncoderDetector()
        if os.path.exists("autoencoder_best.pth"):
            detector.load_detector("autoencoder_best.pth")
        else:
            detector.load_detector("autoencoder.pth")
        
    # 3. Simuler Attaque sur Hôpital 3
    print("\n⚠️ Simulation d'attaque sur Hôpital 3...")
    h3_loader = DataLoader(hospital_datasets[2], batch_size=Config.BATCH_SIZE)
    h3_attacked = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model, h3_loader, attack_type='pgd', ratio=0.5
    )
    # Conversion en TensorDataset
    h3_imgs = torch.cat([item[0] for item in h3_attacked])
    h3_lbls = torch.tensor([item[1].item() for item in h3_attacked])
    dataset_h3_attacked = TensorDataset(h3_imgs, h3_lbls)
    
    raw_datasets = [
        hospital_datasets[0],      # Clean
        hospital_datasets[1],      # Clean
        dataset_h3_attacked,       # Attacked
        hospital_datasets[3]       # Clean
    ]
    
    # 4. Filtrage
    fl_ready_datasets = []
    print_header("🛡️ FILTRAGE DES DONNÉES ENTRANTES")
    
    for i, ds in enumerate(raw_datasets):
        print(f"Hôpital {i+1} : Analyse...")
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False)
        clean_data = detector.filter_clean_data(loader)
        
        if clean_data:
            c_imgs = torch.stack([item[0] for item in clean_data])
            c_lbls = torch.stack([item[1] for item in clean_data])
            fl_ready_datasets.append(TensorDataset(c_imgs, c_lbls))
            print(f"  -> Données validées : {len(clean_data)} / {len(ds)}")
        else:
            print(f"  -> 🛑 Tout rejeté (Attaque détectée) !")

    if not fl_ready_datasets:
        print("❌ Erreur: Aucune donnée n'a passé le filtrage.")
        return

    # 5. Federated Learning
    print_header("🌍 DÉMARRAGE DE L'APPRENTISSAGE FÉDÉRÉ")
    global_model = get_model(pretrained=True)
    fed_learning = FederatedLearning(global_model)
    fed_learning.federated_training(fl_ready_datasets, test_loader=test_loader)
    
    final_acc = fed_learning.evaluate_global_model(test_loader)
    
    # Sauvegarde
    save_name = f"global_model_{method}.pth"
    fed_learning.save_global_model(save_name)
    print(f"\n✅ Modèle Global ({method}) terminé et sauvegardé : {save_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True, choices=['train_detectors', 'federated'])
    parser.add_argument('--method', type=str, choices=['supervised', 'autoencoder'], help="Requis pour l'étape 'federated'")
    args = parser.parse_args()

    set_seed()
    
    if args.step == 'train_detectors':
        train_detectors_step()
    elif args.step == 'federated':
        if not args.method:
            print("❌ Erreur: --method requis pour l'étape federated (supervised ou autoencoder)")
            return
        federated_learning_step(args.method)

if __name__ == "__main__":
    main()