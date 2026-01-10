import torch
import torch.nn as nn
from app.config import Config

class AdversarialAttacks:
    """
    Implémentation des attaques adversariales FGSM et PGD
    """
    
    @staticmethod
    def fgsm_attack(model, images, labels, epsilon=Config.EPSILON_FGSM):
        """
        Fast Gradient Sign Method (FGSM)
        
        Args:
            model: Modèle à attaquer
            images: Images d'entrée
            labels: Labels vrais
            epsilon: Magnitude de la perturbation
        
        Returns:
            Images adversariales
        """
        images = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Créer la perturbation
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = images + epsilon * sign_data_grad
        
        # Clipper pour rester dans [0, 1] après dénormalisation
        perturbed_image = torch.clamp(perturbed_image, -3, 3)
        
        return perturbed_image.detach()
    
    @staticmethod
    def pgd_attack(model, images, labels, 
                   epsilon=Config.EPSILON_PGD,
                   alpha=Config.PGD_ALPHA,
                   num_iter=Config.PGD_ITERATIONS):
        """
        Projected Gradient Descent (PGD)
        
        Args:
            model: Modèle à attaquer
            images: Images d'entrée
            labels: Labels vrais
            epsilon: Magnitude maximale de la perturbation
            alpha: Taille du pas
            num_iter: Nombre d'itérations
        
        Returns:
            Images adversariales
        """
        original_images = images.clone().detach()
        
        # Initialiser avec du bruit aléatoire
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, -3, 3)
        
        for i in range(num_iter):
            perturbed_images.requires_grad = True
            
            # Forward pass
            outputs = model(perturbed_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Mettre à jour la perturbation
            with torch.no_grad():
                data_grad = perturbed_images.grad.data
                perturbed_images = perturbed_images + alpha * data_grad.sign()
                
                # Projeter sur la boule epsilon
                perturbation = torch.clamp(
                    perturbed_images - original_images,
                    min=-epsilon,
                    max=epsilon
                )
                perturbed_images = original_images + perturbation
                perturbed_images = torch.clamp(perturbed_images, -3, 3)
        
        return perturbed_images.detach()
    
    @staticmethod
    def generate_adversarial_dataset(model, dataloader, attack_type='fgsm', ratio=0.3):
        """
        Génère un dataset avec des exemples adversariaux
        
        Args:
            model: Modèle à utiliser
            dataloader: DataLoader source
            attack_type: Type d'attaque ('fgsm' ou 'pgd')
            ratio: Proportion d'exemples adversariaux
        
        Returns:
            Liste de tuples (image, label, is_adversarial)
        """
        model.eval()
        adversarial_data = []
        
        attack_fn = AdversarialAttacks.fgsm_attack if attack_type == 'fgsm' else AdversarialAttacks.pgd_attack
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            # Décider quelles images attaquer
            num_to_attack = int(len(images) * ratio)
            attack_indices = torch.randperm(len(images))[:num_to_attack]
            
            for i in range(len(images)):
                if i in attack_indices:
                    # Générer un exemple adversarial
                    adv_image = attack_fn(model, images[i:i+1], labels[i:i+1])
                    adversarial_data.append((adv_image.cpu(), labels[i].cpu(), 1))
                else:
                    # Garder l'image originale
                    adversarial_data.append((images[i:i+1].cpu(), labels[i].cpu(), 0))
        
        return adversarial_data