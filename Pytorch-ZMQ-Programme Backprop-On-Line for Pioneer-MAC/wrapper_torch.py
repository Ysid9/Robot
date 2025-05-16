import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import threading
import json
import atexit
import math
import matplotlib.pyplot as plt

# Importer les modules PyTorch existants
# Assurez-vous que ces fichiers sont dans le même répertoire
from torch_back import PioneerNN
from torch_online import PyTorchOnlineTrainer

# Importer le module de simulation
from zmq_pioneer_simulation import ZMQPioneerSimulation

# Importer le module de monitoring corrigé
from monitoring import RobotMonitorAdapter


class MonitoredPyTorchTrainer(PyTorchOnlineTrainer):
    """
    Version modifiée du PyTorchOnlineTrainer avec monitoring intégré
    """
    
    def __init__(self, robot, nn_model, monitor_adapter=None):
        """
        Initialise l'entraîneur avec monitoring
        
        Args:
            robot: Instance du robot (ZMQPioneerSimulation)
            nn_model: Modèle PyTorch du réseau de neurones
            monitor_adapter: Adaptateur de monitoring (si None, un nouveau sera créé)
        """
        super().__init__(robot, nn_model)
        
        # Créer ou utiliser l'adaptateur de monitoring fourni
        self.monitor = monitor_adapter or RobotMonitorAdapter(self.alpha)
        
        # Variables pour le suivi des données
        self.last_command = [0, 0]
        self.last_gradient = [0, 0]
        self.last_cost = 0
        
    def train(self, target):
        """
        Procédure d'apprentissage en ligne avec monitoring
        
        Args:
            target (list): position cible [x,y,theta]
        """
        # Informer le moniteur de la cible
        self.monitor.set_target(target)
        
        # Le reste est similaire à la méthode originale
        position = self.robot.get_position()
        
        # Calculer l'entrée du réseau (erreur normalisée)
        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - self._theta_s(position[0], position[1])) * self.alpha[2]
        
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
            
            # Sauvegarder la commande pour le monitoring
            self.last_command = command
            
            # Calculer le critère avant de déplacer le robot
            alpha_x = 1/6
            alpha_y = 1/6
            alpha_teta = 1.0/(math.pi)
            
            crit_av = (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                       alpha_y * alpha_y * (position[1] - target[1])**2 + 
                       alpha_teta * alpha_teta * (position[2] - target[2] - 
                                                 self._theta_s(position[0], position[1]))**2)
            
            # Appliquer les commandes au robot
            self.robot.set_motor_velocity(command)
            
            # Mettre à jour le monitoring avec l'état actuel (avant déplacement)
            self.monitor.update(
                position=position,
                wheel_speeds=command,
                gradient=self.last_gradient,
                cost=crit_av
            )
            
            # Attendre un court instant
            time.sleep(0.050)
            
            # Obtenir la nouvelle position du robot
            position = self.robot.get_position()
            
            # Mettre à jour l'entrée du réseau
            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - self._theta_s(position[0], position[1])) * self.alpha[2]
            
            # Calculer le critère après déplacement
            crit_ap = (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                      alpha_y * alpha_y * (position[1] - target[1])**2 + 
                      alpha_teta * alpha_teta * (position[2] - target[2] - 
                                                self._theta_s(position[0], position[1]))**2)
            
            # Sauvegarder le coût pour le monitoring
            self.last_cost = crit_ap
            
            # Apprentissage (si activé)
            if self.training:
                delta_t = (time.time() - debut)
                
                # Calculer le gradient du critère par rapport aux sorties du réseau
                grad = [
                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    -alpha_teta*alpha_teta*(position[2]-target[2]-self._theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R)),

                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    +alpha_teta*alpha_teta*(position[2]-target[2]-self._theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R))
                    ]
                
                # Sauvegarder le gradient pour le monitoring
                self.last_gradient = grad
                
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
    
    def _theta_s(self, x, y):
        """Fonction theta_s utilisée pour le calcul du critère"""
        return math.tanh(10.*x)*math.atan(1.*y)


# Fonction principale pour exécuter l'apprentissage avec monitoring
def run_with_monitoring():
    # Initialize the robot with ZMQ Remote API
    robot = ZMQPioneerSimulation()
    
    # Variable pour le trainer (utile pour le nettoyage)
    trainer = None

    # Register cleanup function to ensure simulation is stopped
    def cleanup():
        print("Cleaning up and stopping simulation...")
        if hasattr(robot, 'cleanup'):
            robot.cleanup()
        
        # S'assurer que le moniteur est arrêté
        if trainer and hasattr(trainer, 'monitor'):
            trainer.monitor.stop_monitoring()
    
    atexit.register(cleanup)

    # Wait a moment to ensure the simulation is fully started
    time.sleep(1)

    # Configuration
    HL_size = 10  # Taille de la couche cachée
    input_size = 3  # x, y, theta
    output_size = 2  # vitesses des roues gauche et droite

    # Créer le moniteur - Ne pas démarrer encore
    monitor = RobotMonitorAdapter()

    # Création du réseau PyTorch
    network = PioneerNN(input_size, HL_size, output_size)
    print("✅ PyTorch neural network created")

    choice = input('Do you want to load previous network? (y/n) --> ')
    if choice == 'y':
        try:
            with open('last_w.json') as fp:
                json_obj = json.load(fp)
            
            # Charger les poids dans le modèle PyTorch
            network.load_weights_from_json(json_obj, HL_size)
            print("✅ Network weights loaded successfully into PyTorch model")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("Starting with a new network")

    # Initialiser le trainer PyTorch avec monitoring
    trainer = MonitoredPyTorchTrainer(robot, network, monitor)

    # Maintenant que l'interface est prête, démarrer le monitoring
    monitor.start_monitoring()

    choice = ''
    while choice!='y' and choice !='n':
        choice = input('Do you want to learn? (y/n) --> ')

    if choice == 'y':
        trainer.training = True
    elif choice == 'n':
        trainer.training = False

    target_input = input("Enter the first target : x y radian --> ")
    target = target_input.split()
    try:
        for i in range(len(target)):
            target[i] = float(target[i])
        print('✅ New target set: [%.2f, %.2f, %.2f]' % (target[0], target[1], target[2]))
    except Exception as e:
        print(f"❌ Error parsing target: {e}")
        print("Using default target [0, 0, 0]")
        target = [0, 0, 0]

    # Mettre à jour la cible dans le moniteur
    monitor.set_target(target)

    continue_running = True
    while(continue_running):
        thread = threading.Thread(target=trainer.train, args=(target,))
        trainer.running = True
        thread.start()

        # Mettre à jour les graphiques pendant l'exécution
        try:
            # Ask for stop running
            input("Press Enter to stop the current training")
            trainer.running = False
            
            # Wait for the thread to finish with a timeout
            thread.join(timeout=5)
            if thread.is_alive():
                print("⚠️ Training thread did not finish in time, but we'll continue")
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping training...")
            trainer.running = False
            thread.join(timeout=5)
        
        # Sauvegarder les résultats de cette session
        try:
            monitor.save_results(f"training_session_{time.strftime('%Y%m%d_%H%M%S')}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        choice = ''
        while choice!='y' and choice !='n':
            choice = input("Do you want to continue ? (y/n) --> ")

        if choice == 'y':
            choice_learning = ''
            while choice_learning != 'y' and choice_learning !='n':
                choice_learning = input('Do you want to learn ? (y/n) --> ')
            if choice_learning =='y':
                trainer.training = True
            elif choice_learning == 'n':
                trainer.training = False
            target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
            try:
                target = target_input.split()
                for i in range(len(target)):
                    target[i] = float(target[i])
                print('✅ New target set: [%.2f, %.2f, %.2f]' % (target[0], target[1], target[2]))
                
                # Mettre à jour la cible dans le moniteur
                monitor.set_target(target)
            except Exception as e:
                print(f"❌ Error parsing target: {e}")
                print("Keeping previous target")
        elif choice == 'n':
            continue_running = False

    # Save the weights
    try:
        # Convertir les poids PyTorch au format JSON compatible
        json_obj = network.save_weights_to_json()
        with open('last_w_torch.json', 'w') as fp:
            json.dump(json_obj, fp)
        print("✅ The PyTorch model weights have been stored in last_w_torch.json")
    except Exception as e:
        print(f"❌ Error saving weights: {e}")

    # Sauvegarder les résultats finaux
    try:
        monitor.save_results(f"final_results_{time.strftime('%Y%m%d_%H%M%S')}")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")

    # Final cleanup 
    cleanup()


# Exécution principale
if __name__ == "__main__":
    run_with_monitoring()