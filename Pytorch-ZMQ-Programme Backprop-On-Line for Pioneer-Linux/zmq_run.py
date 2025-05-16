from zmq_BackProp_Python_v2 import NN
from zmq_pioneer_simulation import ZMQPioneerSimulation
# from rdn import Pioneer # rdn pour ROS avec le pioneer
# import rospy
from zmq_online_trainer import OnlineTrainer
import json
import threading
import atexit
import time

# Initialize the robot with improved ZMQ Remote API
robot = ZMQPioneerSimulation()
# robot = Pioneer(rospy)

# Register cleanup function to ensure simulation is stopped
def cleanup():
    print("Cleaning up and stopping simulation...")
    if hasattr(robot, 'cleanup'):
        robot.cleanup()
atexit.register(cleanup)

# Wait a moment to ensure the simulation is fully started
time.sleep(1)

HL_size = 10  # nbre neurons of Hidden layer
network = NN(3, HL_size, 2)

choice = input('Do you want to load previous network? (y/n) --> ')
if choice == 'y':
    try:
        with open('last_w.json') as fp:
            json_obj = json.load(fp)

        for i in range(3):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j]
        print("✅ Network weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Starting with a new network")

trainer = OnlineTrainer(robot, network)

choice = ''
while choice != 'y' and choice != 'n':
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

continue_running = True
while(continue_running):
    thread = threading.Thread(target=trainer.train, args=(target,))
    trainer.running = True
    thread.start()

    # Ask for stop running
    input("Press Enter to stop the current training")
    trainer.running = False
    
    # Wait for the thread to finish with a timeout
    thread.join(timeout=5)
    if thread.is_alive():
        print("⚠️ Training thread did not finish in time, but we'll continue")
    
    choice = ''
    while choice != 'y' and choice != 'n':
        choice = input("Do you want to continue? (y/n) --> ")

    if choice == 'y':
        choice_learning = ''
        while choice_learning != 'y' and choice_learning != 'n':
            choice_learning = input('Do you want to learn? (y/n) --> ')
        if choice_learning == 'y':
            trainer.training = True
        elif choice_learning == 'n':
            trainer.training = False
        
        target_input = input("Move the robot to the initial point and enter the new target : x y radian --> ")
        try:
            target = target_input.split()
            for i in range(len(target)):
                target[i] = float(target[i])
            print('✅ New target set: [%.2f, %.2f, %.2f]' % (target[0], target[1], target[2]))
        except Exception as e:
            print(f"❌ Error parsing target: {e}")
            print("Keeping previous target")
    elif choice == 'n':
        continue_running = False

# Save the weights
try:
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open('last_w.json', 'w') as fp:
        json.dump(json_obj, fp)
    print("✅ The last weights have been stored in last_w.json")
except Exception as e:
    print(f"❌ Error saving weights: {e}")

# Final cleanup 
cleanup()