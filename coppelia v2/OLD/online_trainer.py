import time
import math


def theta_s(x,y):
    # if x>=0:
    #     return 1*math.atan(1*y)
    #     #return (math.pi)/4*2*x*math.exp(-x*x)
    
    # if x<0:
    #     return 1*math.atan(-1*y)
    #     #return -(math.pi)/4*2*x*math.exp(-x*x)
    
    # return 2/math.pi*math.atan(x)*math.atan(y)
    
    # r = math.sqrt(x**2+y**2)
    # angle = math.atan2(y,x)
    # if x == 0 :
    #     correction = 0
    # else:
    #     correction = (x/abs(x)+1)/2
    # return math.tanh(r)*(angle-correction*math.pi)
    
    # return math.tanh(5.*x)*math.atan(1.*y)
    return math.exp(-y**2) 

class OnlineTrainer:
    def __init__(self, robot, NN, alpha_r):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.alpha_robot = alpha_r

        self.alpha = [1/4,1/4,1/(math.pi)]  # normalition avec limite du monde cartesien = -3m � + 3m
        
        self.monitoring = []
        
    def position_monitoring(self):
        return self.monitoring

    def train(self, target):
        position = self.robot.get_position(self.robot.omniBot)
        
        network_input = [0, 0, 0]
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
        #Teta_t = 0

        while self.running:
            debut = time.time()
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            
            monitor = self.robot.get_position(self.robot.omniBot)
            self.monitoring.append(monitor)
            
            alpha_x = self.alpha_robot[0]
            alpha_y = self.alpha_robot[1]
            alpha_teta = self.alpha_robot[2]
            
            # Compute error and associated cost function
            epsilon_x = position[0]-target[0]
            epsilon_y = position[1]-target[1]
            epsilon_theta = position[2]-target[2] - theta_s(position[0], position[1])
            
            crit_av = alpha_x**2 * epsilon_x**2 + alpha_y**2 * epsilon_y**2 + alpha_teta**2 * epsilon_theta**2
                        
            # crit_av= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) + alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*(position[2]-target[2]-theta_s(position[0], position[1]))  
            
            self.robot.set_motor_velocity(command) # applique vitesses roues instant t,                     
            time.sleep(0.050) # attend delta t
            position = self.robot.get_position(self.robot.omniBot) #  obtient nvlle pos robot instant t+1       
            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
            
            crit_ap = alpha_x**2 * epsilon_x**2 + alpha_y**2 * epsilon_y**2 + alpha_teta**2 * epsilon_theta**2
            # crit_ap= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) + alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*(position[2]-target[2]-theta_s(position[0], position[1])) 

            if self.training:
                delta_t = (time.time()-debut)
                
                # Update the error
                epsilon_x = position[0]-target[0]
                epsilon_y = position[1]-target[1]
                epsilon_theta = position[2]-target[2] - theta_s(position[0], position[1])
                
                # Compute gradient
                grad_x = alpha_x**2 * epsilon_x * delta_t * self.robot.r * math.cos(position(2))
                grad_y = alpha_y**2 * epsilon_y * delta_t * self.robot.r * math.sin(position(2))
                grad_theta = alpha_teta**2 * epsilon_theta * delta_t * self.robot.r / (2*self.robot.R)
                
                grad = [
                    (-1/delta_t)*(grad_x + grad_y - grad_theta),

                    (-1/delta_t)*(grad_x + grad_y + grad_theta)
                    ]

                # grad = [
                #     (-1/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                #     +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                #     -alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R)),

                #     (-1/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                #     +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                #     +alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R))
                #     ]

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap < crit_av) :
                    self.network.backPropagate(grad,0.5, 0.0) # grad, pas d'app, moment
                else :
                  # self.network.random_update(0.001)                                        
                    self.network.backPropagate(grad, 0.5, 0.0)
                
        self.robot.set_motor_velocity([0,0]) # stop  apres arret  du prog d'app
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
             
                
        
        self.running = False
