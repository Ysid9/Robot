import torch
import torch.nn as nn
import time
import math
import numpy as np
import sim

# Définition d'un modèle PyTorch simple
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
        # Ajouter une activation tanh en sortie pour limiter les valeurs entre -1 et 1
        return torch.tanh(x)

# Classe pour tester la connexion avec CoppeliaSim via Legacy Remote API
class VrepPytorchTest:
    def __init__(self):
        # Paramètres similaires à ceux de vrep_pioneer_simulation.py
        self.ip = '127.0.0.1'
        self.port = 19997
        self.scene = './simu.ttt'
        self.gain = 2
        self.initial_position = [3, 3, self.to_rad(45)]

        self.r = 0.096  # rayon des roues
        self.R = 0.267  # demi-distance entre les roues

        print('Démarrage du test PyTorch avec CoppeliaSim')
        sim.simxFinish(-1)  # Fermer toutes les connexions existantes
        self.client_id = sim.simxStart(self.ip, self.port, True, True, 5000, 5)

        if self.client_id != -1:
            print(f'OK Connecté à l\'API distante sur {self.ip}:{self.port}')
            
            # Arrêter toute simulation en cours
            sim.simxStopSimulation(self.client_id, sim.simx_opmode_oneshot_wait)
            time.sleep(1)  # Attendre que la simulation s'arrête complètement

            
            res = sim.simxLoadScene(self.client_id, self.scene, 1, sim.simx_opmode_oneshot_wait)
            if res == sim.simx_return_ok:
                print(f'OK Scène chargée avec succès: {self.scene}')
            else:
                print(f'KO Erreur lors du chargement de la scène: {res}')
            
            res, self.pioneer = sim.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx', sim.simx_opmode_oneshot_wait)
            if res == sim.simx_return_ok:
                print(f'OK Objet Pioneer trouvé')
            else:
                print(f'KO Erreur lors de la récupération de l\'objet Pioneer: {res}')
                
            res, self.left_motor = sim.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_leftMotor', sim.simx_opmode_oneshot_wait)
            res, self.right_motor = sim.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_rightMotor', sim.simx_opmode_oneshot_wait)

            # Vérifier si on peut définir la position initiale
            success = self.set_position(self.initial_position)
            if success:
                print('OK Position initiale définie avec succès')
            else:
                print('KO Erreur lors de la définition de la position initiale')
            
            # Démarrer la simulation
            res = sim.simxStartSimulation(self.client_id, sim.simx_opmode_oneshot_wait)
            if res == sim.simx_return_ok:
                print('OK Simulation démarrée avec succès')
            else:
                print(f'KO Erreur lors du démarrage de la simulation: {res}')
                
        else:
            print(f'KO Impossible de se connecter à {self.ip}:{self.port}')
            print('Assurez-vous que CoppeliaSim est en cours d\'exécution et que la scène est chargée.')

    def to_rad(self, deg):
        return 2 * math.pi * deg / 360

    def to_deg(self, rad):
        return rad * 360 / (2 * math.pi)

    def set_position(self, position):
        """Définir la position (x,y,theta) du robot"""
        res1 = sim.simxSetObjectPosition(self.client_id, self.pioneer, -1, [position[0], position[1], 0.13879], sim.simx_opmode_oneshot_wait)
        res2 = sim.simxSetObjectOrientation(self.client_id, self.pioneer, -1, [0, 0, position[2]], sim.simx_opmode_oneshot_wait)
        return res1 == sim.simx_return_ok and res2 == sim.simx_return_ok

    def get_position(self):
        """Obtenir la position (x,y,theta) du robot"""
        position = []
        res, tmp = sim.simxGetObjectPosition(self.client_id, self.pioneer, -1, sim.simx_opmode_oneshot_wait)
        if res == sim.simx_return_ok:
            position.append(tmp[0])
            position.append(tmp[1])
        else:
            position.append(0)
            position.append(0)
            print(f'KO Erreur lors de la récupération de la position: {res}')

        res, tmp = sim.simxGetObjectOrientation(self.client_id, self.pioneer, -1, sim.simx_opmode_oneshot_wait)
        if res == sim.simx_return_ok:
            position.append(tmp[2])  # en radians
        else:
            position.append(0)
            print(f'KO Erreur lors de la récupération de l\'orientation: {res}')

        return position

    def set_motor_velocity(self, control):
        """Définir une vitesse cible sur les moteurs du pioneer"""
        res1 = sim.simxSetJointTargetVelocity(self.client_id, self.left_motor, self.gain*control[0], sim.simx_opmode_oneshot_wait)
        res2 = sim.simxSetJointTargetVelocity(self.client_id, self.right_motor, self.gain*control[1], sim.simx_opmode_oneshot_wait)
        return res1 == sim.simx_return_ok and res2 == sim.simx_return_ok

    def test_pytorch_connection(self):
        """Tester la connexion entre PyTorch et CoppeliaSim"""
        # Créer un modèle PyTorch avec la même structure que dans votre implémentation
        input_size = 3  # [x, y, theta] normalisés
        hidden_size = 10
        output_size = 2  # [vitesse_roue_gauche, vitesse_roue_droite]
        model = SimpleNN(input_size, hidden_size, output_size)
        
        # Afficher la structure du modèle
        print("\nStructure du modèle PyTorch:")
        print(model)
        
        initial_pos = self.get_position()
        print("\nPosition initiale:", initial_pos)
        
        # Test avec mouvement sinusoïdal prédéfini
        print("\n=== Test avec mouvement prédéfini ===")
        for i in range(50):  # 3 secondes à 100ms par itération
            # Commandes sinusoïdales pour créer un mouvement visible
            left_speed = 0.5 * math.sin(i * 0.2) + 0.5
            right_speed = 0.5 * math.sin(i * 0.2 + math.pi) + 0.5
            
            commands = [left_speed, right_speed]
            success = self.set_motor_velocity(commands)
            
            position = self.get_position()
            if success:
                print(f"Itération {i+1}: Position = [{position[0]:.2f}, {position[1]:.2f}, {self.to_deg(position[2]):.2f}°], Commande = [{commands[0]:.2f}, {commands[1]:.2f}]")
            else:
                print(f"KO Erreur lors de l'envoi des commandes aux moteurs")
            
            time.sleep(0.1)
        
        # Test avec le modèle PyTorch
        print("\n=== Test avec le modèle PyTorch ===")
        # Revenir à la position initiale
        self.set_position(self.initial_position)
        time.sleep(1)  # Attendre que le robot se stabilise
        
        for i in range(50):  # 5 secondes à 100ms par itération
            # Obtenir la position actuelle
            position = self.get_position()
            
            # Normaliser les entrées (comme dans votre code existant)
            alpha = [1/6, 1/6, 1/math.pi]  # Facteurs de normalisation
            inputs = torch.tensor([
                position[0] * alpha[0],
                position[1] * alpha[1],
                position[2] * alpha[2]
            ], dtype=torch.float32)
            
            # Passage forward dans le réseau
            with torch.no_grad():
                outputs = model(inputs)
            
            # Convertir les sorties de PyTorch en commandes pour le robot
            motor_commands = outputs.detach().numpy()
            
            # Appliquer la commande au robot
            success = self.set_motor_velocity(motor_commands)
            
            if success:
                print(f"Itération {i+1}: Position = [{position[0]:.2f}, {position[1]:.2f}, {self.to_deg(position[2]):.2f}°], Commande PyTorch = [{motor_commands[0]:.2f}, {motor_commands[1]:.2f}]")
            else:
                print(f"KO Erreur lors de l'envoi des commandes PyTorch aux moteurs")
            
            # Attendre un peu
            time.sleep(0.1)
        
        # Vérifier si le robot a bougé par rapport à sa position initiale
        final_pos = self.get_position()
        distance_moved = math.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)
        
        print("\n=== Résultats du test ===")
        print(f"Position initiale: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {self.to_deg(initial_pos[2]):.2f}°]")
        print(f"Position finale: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {self.to_deg(final_pos[2]):.2f}°]")
        print(f"Distance parcourue: {distance_moved:.2f} m")
        
        if distance_moved > 0.1:  # Si le robot a bougé d'au moins 10 cm
            print("OK Le robot a bougé! PyTorch est bien connecté à CoppeliaSim.")
        else:
            print("KO Le robot ne semble pas avoir bougé. Vérifiez la connexion.")
        
        # Arrêter les moteurs à la fin
        self.set_motor_velocity([0, 0])
        
        # Arrêter la simulation
        sim.simxStopSimulation(self.client_id, sim.simx_opmode_oneshot_wait)
        time.sleep(1)  # Attendre que la simulation s'arrête complètement

        sim.simxFinish(self.client_id)
        print("\nTest terminé.")

if __name__ == "__main__":
    test = VrepPytorchTest()
    test.test_pytorch_connection()