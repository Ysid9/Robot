import torch
import json
import threading
import atexit
import time
import sys
import os
from zmq_pioneer_simulation import ZMQPioneerSimulation

# Importation de nos nouvelles classes PyTorch
from torch_back import PioneerNN
from torch_online import PyTorchOnlineTrainer

# Importer le module de monitoring modifié
from monitoring import RobotMonitorAdapter

# Définir les dimensions du monde pour le monitoring
WORLD_BOUNDS = (-6, 6, -6, 6)  # x_min, x_max, y_min, y_max

def run_monitored_training():
    """
    Fonction principale pour exécuter l'apprentissage avec monitoring amélioré
    """
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

    # Création du réseau PyTorch
    network = PioneerNN(input_size, HL_size, output_size)
    print("✅ PyTorch neural network created")

    # Créer l'adaptateur de monitoring avec les limites du monde définies
    monitor_adapter = RobotMonitorAdapter(world_bounds=WORLD_BOUNDS)
    
    # Démarrer le monitoring avant de demander les paramètres à l'utilisateur
    print("✅ Starting monitoring interface...")
    monitor_adapter.start_monitoring()

    # Chargement de poids existants
    choice = input('Do you want to load previous network? (y/n) --> ')
    if choice.lower() == 'y':
        try:
            # Essayer d'abord de charger les poids PyTorch
            if os.path.exists('last_w_torch.json'):
                with open('last_w_torch.json') as fp:
                    json_obj = json.load(fp)
                network.load_weights_from_json(json_obj, HL_size)
                print("✅ PyTorch network weights loaded successfully")
            # Sinon, essayer les poids classiques
            elif os.path.exists('last_w.json'):
                with open('last_w.json') as fp:
                    json_obj = json.load(fp)
                network.load_weights_from_json(json_obj, HL_size)
                print("✅ Original network weights loaded successfully")
            else:
                print("⚠️ No weight files found, using random initialization")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("Starting with a new network")

    # Créer l'optimiseur PyTorch
    optimizer = torch.optim.SGD(network.parameters(), lr=0.2, momentum=0)
    
    # Classe d'entraîneur avec monitoring intégré
    class MonitoredTrainer(PyTorchOnlineTrainer):
        def __init__(self, robot, nn_model, monitor):
            super().__init__(robot, nn_model)
            self.monitor = monitor
            self.last_gradient = [0, 0]
        
        def train(self, target):
            """
            Version améliorée de la méthode d'entraînement avec monitoring intégré
            """
            # Informer le moniteur de la cible
            self.monitor.set_target(target)
            
            # Obtenir la position initiale
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
                
                # Calculer le critère avant de déplacer le robot
                alpha_x = 1/6
                alpha_y = 1/6
                alpha_teta = 1.0/(math.pi)
                
                crit_av = (alpha_x * alpha_x * (position[0] - target[0])**2 + 
                           alpha_y * alpha_y * (position[1] - target[1])**2 + 
                           alpha_teta * alpha_teta * (position[2] - target[2] - 
                                                     self._theta_s(position[0], position[1]))**2)
                
                # Mettre à jour le monitoring avant d'appliquer les commandes
                self.monitor.update(
                    position=position,
                    wheel_speeds=command,
                    gradient=self.last_gradient,
                    cost=crit_av
                )
                
                # Appliquer les commandes au robot
                self.robot.set_motor_velocity(command)
                
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
    
    # Initialiser le trainer avec monitoring
    trainer = MonitoredTrainer(robot, network, monitor_adapter)

    choice = ''
    while choice.lower() not in ['y', 'n']:
        choice = input('Do you want to learn? (y/n) --> ')

    if choice.lower() == 'y':
        trainer.training = True
    else:
        trainer.training = False

    # Demander la cible
    target_input = input("Enter the first target : x y radian --> ")
    try:
        target = [float(x) for x in target_input.split()]
        if len(target) != 3:
            raise ValueError("Need exactly 3 values")
        print(f'✅ New target set: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]')
    except Exception as e:
        print(f"❌ Error parsing target: {e}")
        print("Using default target [0, 0, 0]")
        target = [0, 0, 0]

    # Boucle principale d'entraînement
    continue_running = True
    session_count = 0
    
    while continue_running:
        # Incrémenter le compteur de sessions
        session_count += 1
        
        print(f"\n⚙️ Starting training session #{session_count}")
        
        # Créer et démarrer le thread d'entraînement
        thread = threading.Thread(target=trainer.train, args=(target,))
        trainer.running = True
        thread.start()

        try:
            # Attendre l'arrêt par l'utilisateur
            input("Press Enter to stop the current training")
            trainer.running = False
            
            # Attendre la fin du thread avec timeout
            thread.join(timeout=5)
            if thread.is_alive():
                print("⚠️ Training thread did not finish in time, continuing anyway")
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping training...")
            trainer.running = False
            thread.join(timeout=5)
        
        # Sauvegarder les résultats de cette session
        try:
            monitor_adapter.save_results(f"session_{session_count}_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"✅ Results for session #{session_count} saved")
        except Exception as e:
            print(f"❌ Error saving session results: {e}")
        
        # Demander à l'utilisateur s'il veut continuer
        choice = ''
        while choice.lower() not in ['y', 'n']:
            choice = input("Do you want to continue? (y/n) --> ")

        if choice.lower() == 'y':
            # Réinitialiser l'état d'apprentissage si nécessaire
            choice_learning = ''
            while choice_learning.lower() not in ['y', 'n']:
                choice_learning = input('Do you want to learn? (y/n) --> ')
            
            trainer.training = (choice_learning.lower() == 'y')
            
            # Demander une nouvelle cible
            target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
            try:
                target = [float(x) for x in target_input.split()]
                if len(target) != 3:
                    raise ValueError("Need exactly 3 values")
                print(f'✅ New target set: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]')
            except Exception as e:
                print(f"❌ Error parsing target: {e}")
                print("Keeping previous target")
        else:
            continue_running = False

    # Sauvegarder les poids du réseau
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
        monitor_adapter.save_results(f"final_results_{time.strftime('%Y%m%d_%H%M%S')}")
        print("✅ Final results saved")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")

    # Final cleanup 
    cleanup()


if __name__ == "__main__":
    # Vérifier si les modules nécessaires sont importés
    missing_modules = []
    try:
        import matplotlib
        import numpy
    except ImportError as e:
        missing_module = str(e).split("'")[1]
        missing_modules.append(missing_module)
    
    if missing_modules:
        print("❌ Modules manquants détectés: " + ", ".join(missing_modules))
        print("⚙️  Veuillez installer les modules manquants avec:")
        print(f"    pip install {' '.join(missing_modules)}")
        sys.exit(1)
    
    try:
        # Importer math ici pour l'utiliser dans la classe MonitoredTrainer
        import math
        # Démarrer l'entraînement avec monitoring
        run_monitored_training()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur. Nettoyage et sortie...")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()