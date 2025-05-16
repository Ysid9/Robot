import vrep
import math
import numpy as np

def to_rad(deg):
    return 2*math.pi*deg/360

def to_deg(rad):
    return rad*360/(2*math.pi)

class VrepomniBotSimulation:
    def __init__(self, init_pose):

        self.ip = '127.0.0.1'
        self.port = 19997
        self.scene = './simu_omni_nodrift.ttt'
        self.gain = 1.1
        self.initial_position = [init_pose[0], init_pose[1], to_rad(init_pose[2])]

        self.r = 0.053 # wheel radius
        self.R = 0.12 # Half-distance between the wheels

        print('New omniBot simulation started')
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart(self.ip, self.port, True, True, 5000, 5)

        if self.client_id!=-1:
            print ('Connected to remote API server on %s:%s' % (self.ip, self.port))
            res = vrep.simxLoadScene(self.client_id, self.scene, 1, vrep.simx_opmode_oneshot_wait)
            
            # Coordinates of the center of the robot, along the wheels axis
            res, self.omniBot = vrep.simxGetObjectHandle(self.client_id, 'Base_robot', vrep.simx_opmode_oneshot_wait)
            
            # Left and right motors, used to send speed commands
            res, self.right_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_FL', vrep.simx_opmode_oneshot_wait)
            res, self.left_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_RR', vrep.simx_opmode_oneshot_wait)
            
            # Front and rear motors are not used for this application but are left for consistency
            res, self.rear_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_FR', vrep.simx_opmode_oneshot_wait)
            res, self.front_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_RL', vrep.simx_opmode_oneshot_wait)
            
            # Cylinder at the front of the robot, used as reference point for trajectory following
            res, self.ref_point = vrep.simxGetObjectHandle(self.client_id, 'Cylinder', vrep.simx_opmode_oneshot_wait) # Used to compute the coordinates of the front of the robot

            self.set_position(self.initial_position)
            
            vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        else:
            print('Unable to connect to %s:%s' % (self.ip, self.port))
            
    # def __del__(self):
    #    vrep.simxStopSimulation(self.client_id,vrep.simx_opmode_blocking)

    def set_position(self, position):
        """Set the position (x,y,theta) of the robot

        Args:
            position (list): the position [x,y,theta]
        """

        vrep.simxSetObjectPosition(self.client_id, self.omniBot, -1, [position[0], position[1], 0.115], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(self.client_id, self.omniBot, -1, [0, 0, to_deg(position[2])], vrep.simx_opmode_oneshot_wait)
    
    def get_position(self, target_id):
        """Get the position (x,y,theta) of the robot

        Return:
            position (list): the position [x,y,theta]
        """
        position = []
        res, tmp = vrep.simxGetObjectPosition(self.client_id, target_id, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[0])
        position.append(tmp[1])

        res, tmp = vrep.simxGetObjectOrientation(self.client_id, target_id, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[2]) # en radian

        return position
    
    def get_drifting_speed(self):
        ## Get the robot's linear and angular velocity in the world frame
        errorCode, linearVel, angularVel = vrep.simxGetObjectVelocity(self.client_id, self.omniBot, vrep.simx_opmode_blocking)
        if errorCode != 0:
            return 0
            raise RuntimeError("Failed to retrieve robot velocities.")
        
        # Get the robot's position and orientation in the world frame
        errorCode, position = vrep.simxGetObjectPosition(self.client_id, self.omniBot, -1, vrep.simx_opmode_blocking)
        if errorCode != 0:
            raise RuntimeError("Failed to retrieve robot position.")
        
        errorCode, orientation = vrep.simxGetObjectOrientation(self.client_id, self.omniBot, -1, vrep.simx_opmode_blocking)
        if errorCode != 0:
            raise RuntimeError("Failed to retrieve robot orientation.")
    
        # Construct the rotation matrix from the orientation (Euler angles)
        alpha, beta, gamma = orientation  # Extract Euler angles (XYZ convention)
        rotationMatrix = np.array([
            [np.cos(beta) * np.cos(gamma), np.cos(gamma) * np.sin(alpha) * np.sin(beta) - np.cos(alpha) * np.sin(gamma), np.cos(alpha) * np.cos(gamma) * np.sin(beta) + np.sin(alpha) * np.sin(gamma)],
            [np.cos(beta) * np.sin(gamma), np.cos(alpha) * np.cos(gamma) + np.sin(alpha) * np.sin(beta) * np.sin(gamma), -np.cos(gamma) * np.sin(alpha) + np.cos(alpha) * np.sin(beta) * np.sin(gamma)],
            [-np.sin(beta),               np.cos(beta) * np.sin(alpha),                                             np.cos(alpha) * np.cos(beta)]
        ])
        inverseRotation = np.linalg.inv(rotationMatrix)  # Compute the inverse rotation matrix
    
        # Transform the linear velocity to the robot's local frame
        localLinearVel = np.dot(inverseRotation, linearVel)
        
        # Extract the Y-speed in the robot's local frame
        y_speed = localLinearVel[1]  # Index 1 corresponds to the Y-axis
    
        return y_speed

    def set_motor_velocity(self, control):
        """Set a target velocity on the omniBot motors, multiplied by the gain
        defined in self.gain

        Args:
            control(list): the control [left_motor, right_motor]
        """
        vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor, self.gain*control[1], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor, self.gain*control[0], vrep.simx_opmode_oneshot_wait)
        
    def getSimTime(self): # Currently does not work
        errorCode, simTime = vrep.simxGetFloatSignal(self.client_id, "simulationTime", vrep.simx_opmode_blocking)()
        return simTime

    def set_recording_state(self, state):
        """Send a signal to CoppeliaSim to start (1) or stop (0) recording."""
        errorCode = vrep.simxSetIntegerSignal(self.client_id, "recordingCommand", state, vrep.simx_opmode_oneshot)
        if errorCode == 0:
            if state == 1:
                print("🎥 Recording started successfully!")
            else:
                print("🛑 Recording stopped successfully!")
        else:
            print(f"⚠️ Failed to set recording state! Error Code: {errorCode}")

            
    def stop(self):
        vrep.simxStopSimulation(self.client_id,vrep.simx_opmode_blocking)
        vrep.simxFinish(self.client_id)
    