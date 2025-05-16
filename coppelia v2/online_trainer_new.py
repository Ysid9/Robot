import time
import math

class OnlineTrainer:
    def __init__(self, robot, NN, alpha_r, theta_r):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.alpha_robot = alpha_r
        self.theta_function = theta_r

        self.alpha = [1/5,1/5,1/(math.pi)]  # normalisation avec limite du monde cartesien
        
        self.monitoring = []
        self.sim_time = []
        
        self.theta_map = {
            'NONE': lambda x,y: 0.,
            'Y_ATAN': lambda x,y: (x+1e-6)/abs(x+1e-6)*math.atan(1.*y),
            'XY_ATAN': lambda x,y: 2/math.pi*math.atan(x)*math.atan(y),
            'TANH': lambda x,y: math.tanh(math.sqrt(x**2+y**2))*(math.atan2(y,x)-((x+1e-6)/abs(x+1e-6)+1)/2*math.pi),
            'XY_TANH': lambda x,y: math.tanh(2.5*x)*math.atan(10*y),
            'WEIGHTED_ATAN': lambda x,y: math.atan(2.*y/(math.sqrt(x**2+y**2))),
            'ATAN': lambda x,y: math.atan(y/x)
            } 
    
    def theta_s(self, x,y, function = "XY_TANH"):
        return self.theta_map[function](x,y)
        
    def position_monitoring(self):
        return self.monitoring

    def train(self, target):
        position = self.robot.get_position(self.robot.omniBot)
        
        network_input = [0, 0, 0]
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        network_input[2] = (position[2]-target[2]-self.theta_s(position[0], position[1], self.theta_function))*self.alpha[2]
        
        self.monitoring.append(['x','y','theta','e_x','e_y','e_theta','q1','q2','drift']) # Make it easier to post process the data again, accounting for changes

        while self.running:
            debut = time.time()
            # self.sim_time.append(testTime)
            
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            
            monitoring_pose = self.robot.get_position(self.robot.omniBot)
            
            monitoring_drifting = [self.robot.get_drifting_speed()]
            
            alpha_x = self.alpha_robot[0]
            alpha_y = self.alpha_robot[1]
            alpha_theta = self.alpha_robot[2] * math.exp(-2*(position[1]-target[1])**2)
            
            # Compute error and associated cost function
            epsilon_x = position[0]-target[0]
            epsilon_y = position[1]-target[1]
            epsilon_theta = position[2]-target[2] - self.theta_s(position[0], position[1], self.theta_function)
            
            monitoring_error = [epsilon_x, epsilon_y, epsilon_theta]
            
            crit_av = alpha_x**2 * epsilon_x**2 + alpha_y**2 * epsilon_y**2 + alpha_theta**2 * epsilon_theta**2
                                    
            self.robot.set_motor_velocity(command) # applique vitesses roues instant t,  
            monitoring_command  = [command[0]*1.1*0.053, command[1]*1.1*0.053]
            self.monitoring.append(monitoring_pose + monitoring_error + monitoring_command + monitoring_drifting)

            time.sleep(0.050) # attend delta t
            position = self.robot.get_position(self.robot.omniBot) #  obtient nvlle pos robot instant t+1       
            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[2] = (position[2]-target[2]-self.theta_s(position[0], position[1], self.theta_function))*self.alpha[2]
            
            crit_ap = alpha_x**2 * epsilon_x**2 + alpha_y**2 * epsilon_y**2 + alpha_theta**2 * epsilon_theta**2

            if self.training:
                delta_t = (time.time()-debut)
                
                # Update the error
                epsilon_x = position[0]-target[0]
                epsilon_y = position[1]-target[1]
                epsilon_theta = position[2]-target[2] - self.theta_s(position[0], position[1], self.theta_function)
                
                # Compute gradient
                grad_x = alpha_x**2 * epsilon_x * delta_t * self.robot.r * math.cos(position[2])
                grad_y = alpha_y**2 * epsilon_y * delta_t * self.robot.r * math.sin(position[2])
                grad_theta = alpha_theta**2 * epsilon_theta * delta_t * self.robot.r / (2*self.robot.R)
                
                grad = [
                    (-1/delta_t)*(grad_x + grad_y - grad_theta),

                    (-1/delta_t)*(grad_x + grad_y + grad_theta)
                    ]

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap < crit_av) :
                    self.network.backPropagate(grad,0.5, 0.0) # grad, pas d'app, moment
                else :
                  # self.network.random_update(0.001)                                        
                    self.network.backPropagate(grad, 0.5, 0.0)
                
        self.robot.set_motor_velocity([0,0]) # stop  apres arret  du prog d'app       
        self.running = False
