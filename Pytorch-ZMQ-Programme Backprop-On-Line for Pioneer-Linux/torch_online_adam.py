import time
import math
import torch
import numpy as np

def theta_s(x, y):
    return math.tanh(10.*x)*math.atan(1.*y)

class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model):
        """
        Args:
            robot (Robot): instance du robot suivant le modèle de ZMQPioneerSimulation
            nn_model (PioneerNN): modèle PyTorch du réseau de neurones
        """
        self.robot = robot
        self.network = nn_model
        
        # Facteurs de normalisation (identiques à l'implémentation originale)
        self.alpha = [1/6, 1/6, 1/(math.pi)]
        
        # État de l'apprentissage
        self.running = False
        self.training = False
        
        # Création de l'optimiseur Adam au lieu de SGD
        # Adam utilise des taux d'apprentissage adaptatifs avec des estimations des moments
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                         lr=0.01,  # Taux d'apprentissage plus faible pour Adam
                                         betas=(0.9, 0.999),  # Facteurs de décroissance pour les moments
                                         eps=1e-8,  # Terme de stabilité numérique
                                         weight_decay=0)  # Pas de régularisation L2
    
    def train(self, target):
        """
        Procédure d'apprentissage en ligne
        
        Args:
            target (list): position cible [x,y,theta]
        """
        # Obtenir la position actuelle du robot
        position = self.robot.get_position()
        
        # Calculer l'entrée du réseau (erreur normalisée)
        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]
        
        # Boucle d'apprentissage
        while self.running:
            # Mesurer le temps de début pour le calcul du delta_t
            debut = time.time()
            
            # Forward pass - obtenir les commandes de vitesse des roues
            input_tensor = torch.tensor(network_input, dtype=torch.float32)
            if self.training:
                # En mode apprentissage, nous voulons les gradients
                input_tensor.requires_grad_(True)
                command = self.network(input_tensor).tolist()
            else:
                # En mode évaluation, pas besoin de gradients
                with torch.no_grad():
                    command = self.network(input_tensor).tolist()
            
            # Calculer le critère avant de déplacer le robot
            alpha_x = 1/6
            alpha_y = 1/6
            alpha_teta = 1.0/(math.pi)
            
            crit_av = (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                       alpha_y * alpha_y * (position[1] - target[1])**2 + 
                       alpha_teta * alpha_teta * (position[2] - target[2] - 
                                                 theta_s(position[0], position[1]))**2)
            
            # Appliquer les commandes au robot
            self.robot.set_motor_velocity(command)
            
            # Attendre un court instant
            time.sleep(0.050)
            
            # Obtenir la nouvelle position du robot
            position = self.robot.get_position()
            
            # Mettre à jour l'entrée du réseau
            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]
            
            # Calculer le critère après déplacement
            crit_ap = (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                      alpha_y * alpha_y * (position[1] - target[1])**2 + 
                      alpha_teta * alpha_teta * (position[2] - target[2] - 
                                                theta_s(position[0], position[1]))**2)
            
            # Apprentissage (si activé)
            if self.training:
                delta_t = (time.time() - debut)
                
                # Calculer le gradient du critère par rapport aux sorties du réseau
                grad = [
                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    -alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R)),

                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    +alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R))
                    ]
                
                # Stratégie d'apprentissage en fonction de l'évolution du critère
                if crit_ap <= crit_av:
                    # Effectuer une étape d'apprentissage
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, 0.2, 0)
                else:
                    # Alternative si le critère ne s'améliore pas
                    # On peut soit ajouter du bruit soit quand même apprendre
                    self.optimizer.zero_grad()
                    
                    # Convertir le gradient
                    grad_tensor = torch.tensor(grad, dtype=torch.float32)
                    
                    # Effectuer une étape d'apprentissage personnalisée
                    self.manual_backward(input_tensor, grad_tensor, 0.2, 0)
        
        # Arrêter le robot à la fin de l'apprentissage
        self.robot.set_motor_velocity([0, 0])
        self.running = False
    
    def manual_backward(self, inputs, grad_tensor, learning_rate, momentum):
        """
        Effectue manuellement une étape de rétropropagation avec le gradient fourni
        
        Args:
            inputs: les entrées du réseau
            grad_tensor: le gradient du critère par rapport aux sorties
            learning_rate: le taux d'apprentissage
            momentum: le facteur de momentum
        """
        # Réinitialiser les gradients
        self.optimizer.zero_grad()
        
        # Forward pass pour établir le graphe de calcul
        outputs = self.network(inputs)
        
        # Rétropropager le gradient directement
        # C'est ici que nous connectons notre gradient externe au graphe PyTorch
        outputs.backward(gradient=-grad_tensor)
        
        # L'optimiseur Adam gère automatiquement les moments et les taux d'apprentissage adaptatifs
        self.optimizer.step()