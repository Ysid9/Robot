import torch
import json
import threading
import atexit
import time
from pytorch_pioneer_simulation import ZMQPioneerSimulation
from pytorch_BackProp_v1 import NN  # Pour compatibilité et comparaison

# Importation de nos nouvelles classes PyTorch
from pytorch_BackProp_v1 import PioneerNN
from pytorch_online_trainer import PyTorchOnlineTrainer

# Initialize the robot with ZMQ Remote API
robot = ZMQPioneerSimulation()

# Register cleanup function to ensure simulation is stopped
def cleanup():
    print("Cleaning up and stopping simulation...")
    if hasattr(robot, 'cleanup'):
        robot.cleanup()
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

# Créer l'optimiseur PyTorch (pas utilisé directement mais utile pour l'état)
optimizer = torch.optim.SGD(network.parameters(), lr=0.2, momentum=0)
network.optimizer = optimizer

# Initialiser le trainer PyTorch
trainer = PyTorchOnlineTrainer(robot, network)

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
    # Convertir les poids PyTorch au format JSON compatible
    json_obj = network.save_weights_to_json()
    with open('last_w_torch.json', 'w') as fp:
        json.dump(json_obj, fp)
    print("✅ The PyTorch model weights have been stored in last_w_torch.json")
except Exception as e:
    print(f"❌ Error saving weights: {e}")

# Final cleanup 
cleanup()