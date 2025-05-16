import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import time
import math
import matplotlib
matplotlib.use('TkAgg')  # Utilisation de TkAgg pour éviter les problèmes dans les threads

def theta_s(x, y):
    return math.tanh(10.*x)*math.atan(1.*y)

class RobotVisualizer:
    def __init__(self):
        # Configuration figure principale avec sous-graphiques
        self.fig = plt.figure(figsize=(12, 10))
        
        # Sous-graphique 1: Trajectoire du robot
        self.ax_trajectory = self.fig.add_subplot(221)
        self.ax_trajectory.set_title('Trajectoire du Robot')
        self.ax_trajectory.set_xlabel('X (m)')
        self.ax_trajectory.set_ylabel('Y (m)')
        self.ax_trajectory.grid(True)
        self.ax_trajectory.set_xlim(-3, 3)
        self.ax_trajectory.set_ylim(-3, 3)
        
        # Sous-graphique 2: Évolution de la fonction de coût
        self.ax_cost = self.fig.add_subplot(222)
        self.ax_cost.set_title('Fonction de Coût')
        self.ax_cost.set_xlabel('Itération')
        self.ax_cost.set_ylabel('Coût')
        self.ax_cost.grid(True)
        
        # Sous-graphique 3: Évolution du gradient
        self.ax_gradient = self.fig.add_subplot(223)
        self.ax_gradient.set_title('Norme du Gradient')
        self.ax_gradient.set_xlabel('Itération')
        self.ax_gradient.set_ylabel('||Gradient||')
        self.ax_gradient.grid(True)
        
        # Sous-graphique 4: Évolution des sorties (x, y, theta)
        self.ax_outputs = self.fig.add_subplot(224)
        self.ax_outputs.set_title('Position vs Cible')
        self.ax_outputs.set_xlabel('Itération')
        self.ax_outputs.set_ylabel('Valeur')
        self.ax_outputs.grid(True)
        
        # Initialisation des données à tracer
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.theta_data = []
        self.cost_data = []
        self.gradient_norm_data = []
        self.target_x = 0
        self.target_y = 0
        self.target_theta = 0
        
        # Objets pour les tracés
        self.trajectory_line, = self.ax_trajectory.plot([], [], 'b-', label='Trajectoire')
        self.robot_point, = self.ax_trajectory.plot([], [], 'ro', markersize=8, label='Robot')
        self.target_point, = self.ax_trajectory.plot([], [], 'g*', markersize=12, label='Cible')
        
        self.cost_line, = self.ax_cost.plot([], [], 'r-', label='Coût')
        self.gradient_line, = self.ax_gradient.plot([], [], 'g-', label='Norme Gradient')
        
        self.x_line, = self.ax_outputs.plot([], [], 'r-', label='X')
        self.y_line, = self.ax_outputs.plot([], [], 'g-', label='Y')
        self.theta_line, = self.ax_outputs.plot([], [], 'b-', label='Theta')
        
        # Ajout des légendes
        self.ax_trajectory.legend()
        self.ax_outputs.legend()
        
        # Variables pour contrôler l'exécution
        self.running = False
        self.iteration = 0
        self.lock = threading.Lock()
        self.fig.tight_layout()
        
    def set_target(self, x, y, theta):
        """Définir la position cible"""
        with self.lock:
            self.target_x = x
            self.target_y = y
            self.target_theta = theta
            self.target_point.set_data([x], [y])
    
    def update_position(self, x, y, theta, cost=None, gradient=None):
        """Mettre à jour la position du robot et les métriques associées"""
        with self.lock:
            self.time_data.append(self.iteration)
            self.x_data.append(x)
            self.y_data.append(y)
            self.theta_data.append(theta)
            
            if cost is not None:
                self.cost_data.append(cost)
            else:
                # Si pas de coût fourni, calculer une approximation
                if len(self.x_data) > 0:
                    cost = (x - self.target_x)**2 + (y - self.target_y)**2 + (theta - self.target_theta - theta_s(x, y))**2
                    self.cost_data.append(cost)
            
            if gradient is not None and len(gradient) >= 2:
                # Calculer la norme du gradient
                gradient_norm = np.sqrt(gradient[0]**2 + gradient[1]**2)
                self.gradient_norm_data.append(gradient_norm)
            else:
                # Si pas de gradient fourni, ajouter une valeur factice
                self.gradient_norm_data.append(0)
            
            self.iteration += 1
    
    def update_plot(self, frame):
        """Fonction de mise à jour pour l'animation"""
        with self.lock:
            if len(self.x_data) == 0:
                return (self.trajectory_line, self.robot_point, self.cost_line, 
                        self.gradient_line, self.x_line, self.y_line, self.theta_line)
            
            # Mise à jour de la trajectoire
            self.trajectory_line.set_data(self.x_data, self.y_data)
            self.robot_point.set_data([self.x_data[-1]], [self.y_data[-1]])
            
            # Mise à jour du coût
            self.cost_line.set_data(self.time_data, self.cost_data)
            if len(self.cost_data) > 1:
                self.ax_cost.set_ylim(0, max(self.cost_data) * 1.1)
                self.ax_cost.set_xlim(0, max(self.time_data))
            
            # Mise à jour du gradient
            self.gradient_line.set_data(self.time_data, self.gradient_norm_data)
            if len(self.gradient_norm_data) > 1:
                self.ax_gradient.set_ylim(0, max(self.gradient_norm_data) * 1.1)
                self.ax_gradient.set_xlim(0, max(self.time_data))
            
            # Mise à jour des sorties
            self.x_line.set_data(self.time_data, self.x_data)
            self.y_line.set_data(self.time_data, self.y_data)
            self.theta_line.set_data(self.time_data, self.theta_data)
            
            if len(self.time_data) > 1:
                self.ax_outputs.set_xlim(0, max(self.time_data))
                min_val = min(min(self.x_data), min(self.y_data), min(self.theta_data))
                max_val = max(max(self.x_data), max(self.y_data), max(self.theta_data))
                self.ax_outputs.set_ylim(min_val - 0.5, max_val + 0.5)
        
        return (self.trajectory_line, self.robot_point, self.cost_line, 
                self.gradient_line, self.x_line, self.y_line, self.theta_line)
    
    def start(self):
        """Démarrer la visualisation en mode interactif"""
        self.running = True
        # Activer le mode interactif
        plt.ion()
        # Afficher la figure
        plt.show(block=False)
        
        # Lancer un thread pour mettre à jour périodiquement
        def update_loop():
            while self.running:
                self.update_plot(None)
                plt.pause(0.1)  # Pause pour rafraîchir l'affichage
                
        self.visualization_thread = threading.Thread(target=update_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def stop(self):
        """Arrêter la visualisation"""
        self.running = False
        plt.close(self.fig)
    
    def reset(self):
        """Réinitialiser les données"""
        with self.lock:
            self.time_data = []
            self.x_data = []
            self.y_data = []
            self.theta_data = []
            self.cost_data = []
            self.gradient_norm_data = []
            self.iteration = 0
