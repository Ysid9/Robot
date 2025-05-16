import math
import time
import sys
import os

# Import the ZMQ API for CoppeliaSim
try:
    import coppeliasim_zmqremoteapi_client as zmq
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    print("✅ Successfully imported coppeliasim_zmqremoteapi_client")
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'zmqRemoteApi/asyncio'))
        from zmqRemoteApi import RemoteAPIClient
        print("✅ Successfully imported zmqRemoteApi")
    except ImportError:
        print("❌ Failed to import ZMQ Remote API client. Please install it with: pip install coppeliasim-zmqremoteapi-client")
        sys.exit(1)

def to_rad(deg):
    return 2*math.pi*deg/360

def to_deg(rad):
    return rad*360/(2*math.pi)

class ZMQPioneerSimulation:
    def __init__(self):
        # Find the directory this script is in
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Use that to create an absolute path to the scene file
        self.scene_path = os.path.join(current_dir, 'simu.ttt')

        self.gain = 2
        self.initial_position = [3, 3, to_rad(45)]

        self.r = 0.096  # wheel radius
        self.R = 0.267  # demi-distance entre les roues

        print('New pioneer simulation with ZMQ Remote API started')
        
        # Initialize object handles
        self.pioneer = None
        self.left_motor = None
        self.right_motor = None
        
        # Create ZMQ client connection
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        print('✅ ZMQ API initialized')
        
        # Stop any previous simulation that might be running
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.1)
            
        # Load the scene
        try:
            self.sim.loadScene(self.scene_path)
            print(f'✅ Scene loaded successfully: {self.scene_path}')
            time.sleep(1)  # Wait for the scene to load
            
            # Get the robot objects
            self.pioneer = self.sim.getObject('/Pioneer_p3dx')
            print(f"✅ Found robot")
            
            # Find motors in the hierarchy
            children = self.sim.getObjectsInTree(self.pioneer)
            for child in children:
                child_name = self.sim.getObjectName(child)
                if 'left' in child_name.lower() and 'motor' in child_name.lower():
                    self.left_motor = child
                    print(f"✅ Found left motor: {child_name}")
                elif 'right' in child_name.lower() and 'motor' in child_name.lower():
                    self.right_motor = child
                    print(f"✅ Found right motor: {child_name}")
            
            # Check if all objects were found
            if self.pioneer is not None and self.left_motor is not None and self.right_motor is not None:
                print("✅ All robot objects found")
                
                # Set initial position
                success = self.set_position(self.initial_position)
                if success:
                    print('✅ Initial position set successfully')
                
                # Start the simulation
                self.sim.startSimulation()
                print('✅ Simulation started successfully')
            else:
                print("❌ Some robot objects were not found")
                if self.pioneer is None:
                    print("   - Missing: Pioneer_p3dx")
                if self.left_motor is None:
                    print("   - Missing: left motor")
                if self.right_motor is None:
                    print("   - Missing: right motor")
                
        except Exception as e:
            print(f'❌ Error loading scene: {str(e)}')
            sys.exit(1)

    def set_position(self, position):
        """Set the position (x,y,theta) of the robot

        Args:
            position (list): the position [x,y,theta]
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.sim.setObjectPosition(self.pioneer, -1, [position[0], position[1], 0.13879])
            self.sim.setObjectOrientation(self.pioneer, -1, [0, 0, position[2]])
            return True
        except Exception as e:
            print(f'❌ Error setting position: {str(e)}')
            return False

    def get_position(self):
        """Get the position (x,y,theta) of the robot

        Return:
            position (list): the position [x,y,theta]
        """
        try:
            pos = self.sim.getObjectPosition(self.pioneer, -1)
            ori = self.sim.getObjectOrientation(self.pioneer, -1)
            return [pos[0], pos[1], ori[2]]
        except Exception as e:
            print(f'❌ Error getting position: {str(e)}')
            return [0, 0, 0]

    def set_motor_velocity(self, control):
        """Set a target velocity on the pioneer motors, multiplied by the gain
        defined in self.gain

        Args:
            control(list): the control [left_motor, right_motor]
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.sim.setJointTargetVelocity(self.left_motor, self.gain * control[0])
            self.sim.setJointTargetVelocity(self.right_motor, self.gain * control[1])
            return True
        except Exception as e:
            print(f'❌ Error setting motor velocities: {str(e)}')
            return False

    def cleanup(self):
        """Stop the simulation and close the connection"""
        try:
            self.sim.stopSimulation()
            print("✅ Simulation stopped")
            # No explicit close needed for the ZMQ client
        except Exception as e:
            print(f'❌ Error during cleanup: {str(e)}')