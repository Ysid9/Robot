from BackProp_Python_v2 import NN
from vrep_omnibot_simulation import VrepomniBotSimulation
from online_trainer_new import OnlineTrainer
import json
import threading
import math
import numpy as np
import RobotMonitoring

############################################## Init ##############################################

HL_size= 30 # Number of neurons in the hidden layer
network = NN(3, HL_size, 2) # Number of inputs (x,y,theta), size of hidden layer, number of outputs (speeds of motors)

alpha = [0.95, 0.98, 1.1/math.pi] # Weights in the cost function (x, y and theta)
theta_s_in = "XY_TANH" # Function that is used to add a skew theta in the training process (removes singularity in x=0)

initial_pose = [-0.5,0.5,-45] # x, y, theta(degrees)
robot = VrepomniBotSimulation(initial_pose)
robot.set_motor_velocity([0,0])

############################################## Main ##############################################
if __name__ == "__main__":
    
    choice = input('Do you want to load previous network? (y/n) --> ')
    
    if choice == 'y':
        with open('last_w.json') as fp:
            json_obj = json.load(fp)
    
        for i in range(3):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j]
    
    trainer = OnlineTrainer(robot, network, alpha, theta_s_in)
    
    choice = ''
    while choice!='y' and choice !='n':
        choice = input('Do you want to learn? (y/n) --> ')
    
    if choice == 'y':
        trainer.training = True
    elif choice == 'n':
        trainer.training = False
    
    target = input("Enter the first target : x y radian --> ")
    target = target.split()
    for i in range(len(target)):
        target[i] = float(target[i])
    print('New target : [%d, %d, %d]'%(target[0], target[1], target[2]))
    
    robot.set_motor_velocity([1,1])
    
    ############################################## Train ##############################################
    continue_running = True
    while(continue_running):
    
        thread = threading.Thread(target=trainer.train, args=(target,))
        trainer.running = True
        thread.start()
    
        #Ask to stop running
        input("Press Enter to stop the current training")
        trainer.running = False
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
            target = input("Move the robot to the initial point and enter the new target : x y radian --> ")
            target = target.split()
            for i in range(len(target)):
                target[i] = float(target[i])
            print('New target : [%d, %d, %d]'%(target[0], target[1], target[2]))
        elif choice == 'n':
            continue_running = False
            
        robot.stop()
    
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open('last_w.json', 'w') as fp:
        json.dump(json_obj, fp)
    
    print("The last weights have been stored in last_w.json")
    
    # time = trainer.sim_time

    
    ############################################## Display results ##############################################
    
    ask_display = ''
    while ask_display!='y' and ask_display !='n':
        ask_display = input("Do you wish to record the data ? (y/n) --> ")
    
    if ask_display == 'y':
        # Iteration is used to automatically keep a track record of experiments and properly order the files
        iteration = np.load('iteration.npy') #edit manually with "np.save('iteration',desired_value)"
        iteration += 1 
        np.save('iteration',iteration)
        
        ###### Parameters ######
        layer_info = "hidden_layers: " + str(HL_size)
        theta_s_info = "theta function: " + theta_s_in
        training_info = "training: " + str(trainer.training)
        title_info = "\n(" + layer_info + ", " + theta_s_info + ", " + training_info + ")"
        
        monitoring = trainer.position_monitoring() # Monitoring data (x, y , theta, q1, q2, drift)
        
        RobotMonitoring.visualize_monitoring(monitoring, target, title_info, str(iteration)) # Creates and saves monitoring plots + raw data
        
        # Save the weights
        if trainer.training==True: # Prevents unnecessary saving of the weights as they are only modified during training
            weight_name = str(iteration) + "_w.json"
            path = "Saved_weights/" + weight_name
            with open(path, 'w') as fp:
                json.dump(json_obj, fp)
            print("Saved weights.")