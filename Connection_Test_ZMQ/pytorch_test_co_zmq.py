import torch
import torch.nn as nn
import time
import math
import numpy as np
import sys
import os

# Import de l'API ZMQ pour CoppeliaSim
try:
    import coppeliasim_zmqremoteapi_client as zmq
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    print("✅ Module coppeliasim_zmqremoteapi_client importé avec succès")
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'zmqRemoteApi/asyncio'))
    from zmqRemoteApi import RemoteAPIClient
    print("✅ Module zmqRemoteApi importé avec succès")

# Définition du modèle PyTorch simple
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialisation des poids pour générer un mouvement plus visible
        torch.nn.init.uniform_(self.fc1.weight, -0.5, 0.5)
        torch.nn.init.uniform_(self.fc2.weight, -0.5, 0.5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return torch.tanh(x)  # Limiter les valeurs entre -1 et 1

class VrepPytorchZmqTest:
    def __init__(self):
        # Paramètres
        self.scene_path = os.path.abspath('./simu.ttt')
        self.gain = 2  # Augmenter le gain pour obtenir plus de mouvement
        self.initial_position = [3, 3, self.to_rad(45)]
        self.r = 0.096  # rayon des roues
        self.R = 0.267  # demi-distance entre les roues
        
        print(f"Chemin de la scène: {self.scene_path}")
        print('Démarrage du test PyTorch avec CoppeliaSim via ZMQ Remote API')
        
        # Initialiser les attributs
        self.pioneer = None
        self.left_motor = None
        self.right_motor = None
        self.debug_mode = False  # Activer pour plus de logs
        
        # Créer un client ZMQ Remote API
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        print('✅ API ZMQ initialisée')
        
        # Arrêter une simulation potentiellement en cours
        self.stop_simulation()
        
        # Charger la scène et configurer les objets
        self.setup_scene()
    
    def setup_scene(self):
        """Charger la scène et configurer les objets du robot"""
        try:
            self.sim.loadScene(self.scene_path)
            print(f'✅ Scène chargée avec succès: {self.scene_path}')
            time.sleep(1)  # Attendre que la scène soit chargée
            
            # Obtenir l'objet du robot
            self.pioneer = self.sim.getObject('/Pioneer_p3dx')
            print(f"✅ Robot trouvé")
            
            # Récupérer les moteurs directement par leur nom
            self.find_motors()
            
            # Vérifier si tous les objets ont été trouvés
            if self.pioneer is not None and self.left_motor is not None and self.right_motor is not None:
                print("✅ Tous les objets du robot ont été trouvés")
                
                # Définir la position initiale
                success = self.set_position(self.initial_position)
                if success:
                    print('✅ Position initiale définie avec succès')
                
                # Démarrer la simulation
                self.sim.startSimulation()
                print('✅ Simulation démarrée avec succès')
            else:
                print("❌ Certains objets du robot n'ont pas été trouvés")
                
        except Exception as e:
            print(f'❌ Erreur lors du chargement de la scène: {str(e)}')
    
    def find_motors(self):
        """Trouver les moteurs du robot"""
        try:
            # Essayer d'abord avec le chemin complet
            try:
                self.left_motor = self.sim.getObject('/Pioneer_p3dx/leftMotor')
                print(f"✅ Moteur gauche trouvé: /Pioneer_p3dx/leftMotor")
            except Exception:
                print("❌ Essai alternatif pour le moteur gauche...")
                try:
                    self.left_motor = self.sim.getObject('/Pioneer_p3dx_leftMotor')
                    print(f"✅ Moteur gauche trouvé: /Pioneer_p3dx_leftMotor")
                except Exception as e:
                    print(f"❌ Impossible de trouver le moteur gauche: {str(e)}")
                    
                    # Chercher tous les objets avec "left" et "motor" dans leur nom
                    all_objects = self.sim.getObjects()
                    for obj in all_objects:
                        try:
                            name = self.sim.getObjectName(obj)
                            if "left" in name.lower() and "motor" in name.lower():
                                print(f"Objet possible trouvé: {name}")
                                self.left_motor = obj
                                print(f"✅ Moteur gauche trouvé par recherche: {name}")
                                break
                        except:
                            pass
            
            try:
                self.right_motor = self.sim.getObject('/Pioneer_p3dx/rightMotor')
                print(f"✅ Moteur droit trouvé: /Pioneer_p3dx/rightMotor")
            except Exception:
                print("❌ Essai alternatif pour le moteur droit...")
                try:
                    self.right_motor = self.sim.getObject('/Pioneer_p3dx_rightMotor')
                    print(f"✅ Moteur droit trouvé: /Pioneer_p3dx_rightMotor")
                except Exception as e:
                    print(f"❌ Impossible de trouver le moteur droit: {str(e)}")
                    
                    # Chercher tous les objets avec "right" et "motor" dans leur nom
                    all_objects = self.sim.getObjects()
                    for obj in all_objects:
                        try:
                            name = self.sim.getObjectName(obj)
                            if "right" in name.lower() and "motor" in name.lower():
                                print(f"Objet possible trouvé: {name}")
                                self.right_motor = obj
                                print(f"✅ Moteur droit trouvé par recherche: {name}")
                                break
                        except:
                            pass
                            
            # Imprimer la hiérarchie des objets pour aider au débogage
            if self.debug_mode:
                print("\nHiérarchie du robot:")
                all_objects = self.sim.getObjectsInTree(self.pioneer)
                for obj in all_objects:
                    try:
                        name = self.sim.getObjectName(obj)
                        print(f"  - Objet: {name}")
                    except:
                        print(f"  - Objet sans nom")
        
        except Exception as e:
            print(f"❌ Erreur lors de la recherche des moteurs: {str(e)}")
    
    def stop_simulation(self):
        """Arrêter la simulation en cours s'il y en a une"""
        try:
            if self.sim.getSimulationState() != self.sim.simulation_stopped:
                self.sim.stopSimulation()
                while self.sim.getSimulationState() != self.sim.simulation_stopped:
                    time.sleep(0.1)
                print("✅ Simulation précédente arrêtée")
        except Exception as e:
            print(f"❌ Erreur lors de l'arrêt de la simulation: {str(e)}")

    def to_rad(self, deg):
        return 2 * math.pi * deg / 360
        
    def to_deg(self, rad):
        return rad * 360 / (2 * math.pi)
        
    def set_position(self, position):
        """Définir la position (x,y,theta) du robot"""
        try:
            self.sim.setObjectPosition(self.pioneer, -1, [position[0], position[1], 0.13879])
            self.sim.setObjectOrientation(self.pioneer, -1, [0, 0, position[2]])
            return True
        except Exception as e:
            print(f'❌ Erreur lors de la définition de la position: {str(e)}')
            return False
            
    def get_position(self):
        """Obtenir la position (x,y,theta) du robot"""
        try:
            pos = self.sim.getObjectPosition(self.pioneer, -1)
            ori = self.sim.getObjectOrientation(self.pioneer, -1)
            return [pos[0], pos[1], ori[2]]
        except Exception as e:
            print(f'❌ Erreur lors de la récupération de la position: {str(e)}')
            return [0, 0, 0]
            
    def set_motor_velocity(self, control):
        """Définir une vitesse cible sur les moteurs du pioneer"""
        try:
            # Pour débogage: afficher les objets moteurs
            if self.debug_mode:
                print(f"Motor objects - Left: {self.left_motor}, Right: {self.right_motor}")
                
            # Appliquer un gain plus élevé pour mieux voir le mouvement
            self.sim.setJointTargetVelocity(self.left_motor, self.gain * control[0])
            self.sim.setJointTargetVelocity(self.right_motor, self.gain * control[1])
            
            # Vérifier que les vitesses ont été appliquées
            if self.debug_mode:
                left_vel = self.sim.getJointTargetVelocity(self.left_motor)
                right_vel = self.sim.getJointTargetVelocity(self.right_motor)
                print(f"Vitesses appliquées: {left_vel}, {right_vel}")
                
            return True
        except Exception as e:
            print(f'❌ Erreur lors de la définition des vitesses des moteurs: {str(e)}')
            return False
            
    def test_pytorch_connection(self):
        """Tester la connexion entre PyTorch et CoppeliaSim"""
        # Vérifier si les objets sont correctement initialisés
        if self.pioneer is None or self.left_motor is None or self.right_motor is None:
            print("❌ Les objets du robot ne sont pas correctement initialisés. Annulation du test.")
            return
            
        # Créer un modèle PyTorch
        input_size = 3  # [x, y, theta] normalisés
        hidden_size = 10
        output_size = 2  # [vitesse_roue_gauche, vitesse_roue_droite]
        model = SimpleNN(input_size, hidden_size, output_size)
        
        print("\nStructure du modèle PyTorch:")
        print(model)
        
        initial_pos = self.get_position()
        print("\nPosition initiale:", initial_pos)
        
        # Test avec mouvement sinusoïdal prédéfini
        print("\n=== Test avec mouvement prédéfini ===")
        for i in range(30):  # 3 secondes à 100ms par itération
            # Commandes sinusoïdales pour créer un mouvement visible
            # Amplifier le mouvement avec des vitesses plus différentes
            left_speed = 2.0 * math.sin(i * 0.2) + 2.0  # Valeurs entre 0 et 4
            right_speed = 2.0 * math.sin(i * 0.2 + math.pi) + 2.0
            
            commands = [left_speed, right_speed]
            success = self.set_motor_velocity(commands)
            
            position = self.get_position()
            if success:
                print(f"Itération {i+1}: Position = [{position[0]:.2f}, {position[1]:.2f}, {self.to_deg(position[2]):.2f}°], Commande = [{commands[0]:.2f}, {commands[1]:.2f}]")
            
            time.sleep(0.1)
        
        # Test avec le modèle PyTorch
        print("\n=== Test avec le modèle PyTorch ===")
        # Revenir à la position initiale
        self.set_position(self.initial_position)
        time.sleep(1)  # Attendre que le robot se stabilise
        
        for i in range(50):  # 5 secondes à 100ms par itération
            # Obtenir la position actuelle
            position = self.get_position()
            
            # Normaliser les entrées
            alpha = [1/6, 1/6, 1/math.pi]  # Facteurs de normalisation
            inputs = torch.tensor([
                position[0] * alpha[0],
                position[1] * alpha[1], 
                position[2] * alpha[2]
            ], dtype=torch.float32)
            
            # Ajouter de la variabilité aux entrées pour éviter des commandes constantes
            if i % 10 == 0:
                inputs = inputs + torch.randn_like(inputs) * 0.1
                
            # Passage forward dans le réseau
            with torch.no_grad():
                outputs = model(inputs)
            
            # Ajouter du bruit aux sorties pour éviter la stagnation
            # et amplifier les sorties pour générer plus de mouvement
            motor_commands = outputs.detach().numpy() * 3.0  # Amplification
            
            # Ajouter de la variabilité aux commandes
            if i % 5 == 0:
                motor_commands[0] += np.random.normal(0, 0.3)
                motor_commands[1] += np.random.normal(0, 0.3)
            
            # Appliquer la commande au robot
            success = self.set_motor_velocity(motor_commands)
            
            if success:
                print(f"Itération {i+1}: Position = [{position[0]:.2f}, {position[1]:.2f}, {self.to_deg(position[2]):.2f}°], Commande PyTorch = [{motor_commands[0]:.2f}, {motor_commands[1]:.2f}]")
            
            time.sleep(0.1)
        
        # Vérifier si le robot a bougé par rapport à sa position initiale
        final_pos = self.get_position()
        distance_moved = math.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)
        
        print("\n=== Résultats du test ===")
        print(f"Position initiale: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {self.to_deg(initial_pos[2]):.2f}°]")
        print(f"Position finale: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {self.to_deg(final_pos[2]):.2f}°]")
        print(f"Distance parcourue: {distance_moved:.2f} m")
        
        if distance_moved > 0.1:  # Si le robot a bougé d'au moins 10 cm
            print("✅ Le robot a bougé! PyTorch est bien connecté à CoppeliaSim via ZMQ.")
        else:
            print("❌ Le robot ne semble pas avoir bougé. Vérifiez la connexion.")
            print("Essayez d'augmenter le gain ou d'activer le mode debug pour plus d'informations.")
        
        # Arrêter les moteurs et la simulation
        self.set_motor_velocity([0, 0])
        self.sim.stopSimulation()
        print("\nTest terminé.")

if __name__ == "__main__":
    test = VrepPytorchZmqTest()
    test.test_pytorch_connection()