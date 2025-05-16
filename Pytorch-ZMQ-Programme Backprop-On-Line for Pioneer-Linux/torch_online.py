import time
import math
import torch
import torch.nn as nn


def theta_s(x, y):
    """
    Calcul de la fonction θs pour la normalisation de l'angle
    """
    return math.tanh(10. * x) * math.atan(1. * y)


class PyTorchOnlineTrainer:
    def __init__(self, robot, nn_model):
        """
        Args:
            robot: instance du robot (ZMQPioneerSimulation)
            nn_model: modèle PyTorch (PioneerNN)
            lr: taux d'apprentissage
        """
        self.robot = robot
        self.network = nn_model
        # Facteurs de normalisation [x, y, θ]
        self.alpha = [1/6, 1/6, 1/math.pi]
        self.running = False
        self.training = False
        # Optimiseur SGD sans momentum (momentum = 0)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), lr=0.2, momentum=0.0
        )
        # Critère MSE pour injecter un target basé sur le gradient externe
        self.criterion = nn.MSELoss()

    def train(self, target):
        """
        Boucle d'apprentissage en ligne
        Args:
            target: position cible [x, y, θ]
        """
        # Position initiale
        position = self.robot.get_position()
        # Entrée normalisée
        network_input = [
            (position[0] - target[0]) * self.alpha[0],
            (position[1] - target[1]) * self.alpha[1],
            (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]
        ]

        while self.running:
            start_time = time.time()
            # Conversion en tenseur pour PyTorch
            input_tensor = torch.tensor(network_input, dtype=torch.float32, requires_grad=True)
            # Forward pass
            outputs = self.network(input_tensor)  # vecteur [v_gauche, v_droite]
            # Commande envoyée au robot
            command = outputs.detach().tolist()
            self.robot.set_motor_velocity(command)

            # Pause pour la simulation
            time.sleep(0.050)
            # Mise à jour de la position
            position = self.robot.get_position()
            # Mise à jour de l'entrée
            network_input = [
                (position[0] - target[0]) * self.alpha[0],
                (position[1] - target[1]) * self.alpha[1],
                (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]
            ]

            if self.training:
                # Durée de l'étape
                delta_t = time.time() - start_time
                # Calcul du gradient externe du critère par rapport aux sorties
                grad = [
                    (-2/delta_t) * (
                        self.alpha[0]**2 * (position[0] - target[0]) * delta_t * self.robot.r * math.cos(position[2])
                        + self.alpha[1]**2 * (position[1] - target[1]) * delta_t * self.robot.r * math.sin(position[2])
                        - self.alpha[2]**2 * (position[2] - target[2] - theta_s(position[0], position[1])) * delta_t * self.robot.r / (2 * self.robot.R)
                    ),
                    (-2/delta_t) * (
                        self.alpha[0]**2 * (position[0] - target[0]) * delta_t * self.robot.r * math.cos(position[2])
                        + self.alpha[1]**2 * (position[1] - target[1]) * delta_t * self.robot.r * math.sin(position[2])
                        + self.alpha[2]**2 * (position[2] - target[2] - theta_s(position[0], position[1])) * delta_t * self.robot.r / (2 * self.robot.R)
                    )
                ]
                grad_tensor = torch.tensor(grad, dtype=torch.float32)
                # Cible pseudo-supervisée en ajoutant le gradient externe
                target_tensor = outputs.detach() + grad_tensor
                # Optimisation via MSELoss
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, target_tensor)
                loss.backward()
                self.optimizer.step()

        # Arrêt du robot
        self.robot.set_motor_velocity([0.0, 0.0])
        self.running = False
