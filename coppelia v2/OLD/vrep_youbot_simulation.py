import vrep
import math

def to_rad(deg):
    return 2*math.pi*deg/360

def to_deg(rad):
    return rad*360/(2*math.pi)

class VrepomniBotSimulation:
    def __init__(self):

        self.ip = '127.0.0.1'
        self.port = 19997
        self.scene = './simu_omni.ttt'
        self.gain = 1.5
        self.initial_position = [-1.5,1.5,to_rad(-45)]

        self.r = 0.053 # wheel radius
        self.R = 0.12 # demi-distance entre les r

        print('New omniBot simulation started')
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart(self.ip, self.port, True, True, 5000, 5)

        if self.client_id!=-1:
            print ('Connected to remote API server on %s:%s' % (self.ip, self.port))
            res = vrep.simxLoadScene(self.client_id, self.scene, 1, vrep.simx_opmode_oneshot_wait)
            res, self.omniBot = vrep.simxGetObjectHandle(self.client_id, 'Base_robot', vrep.simx_opmode_oneshot_wait)
            res, self.front_left_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_FL', vrep.simx_opmode_oneshot_wait)
            res, self.front_right_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_FR', vrep.simx_opmode_oneshot_wait)
            res, self.rear_left_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_RL', vrep.simx_opmode_oneshot_wait)
            res, self.rear_right_motor = vrep.simxGetObjectHandle(self.client_id, 'rollingJoint_RR', vrep.simx_opmode_oneshot_wait)
            
            res, self.ref_point = vrep.simxGetObjectHandle(self.client_id, 'Cylinder', vrep.simx_opmode_oneshot_wait)

            self.set_position(self.initial_position)
            vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

        else:
            print('Unable to connect to %s:%s' % (self.ip, self.port))

    def set_position(self, position):
        """Set the position (x,y,theta) of the robot

        Args:
            position (list): the position [x,y,theta]
        """

        vrep.simxSetObjectPosition(self.client_id, self.omniBot, -1, [position[0], position[1], 0.115], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(self.client_id, self.omniBot, -1, [0, 0, to_deg(position[2])], vrep.simx_opmode_oneshot_wait)

    def get_position2(self):
        """Get the position (x,y,theta) of the robot

        Return:
            position (list): the position [x,y,theta]
        """
        position = []
        res, tmp = vrep.simxGetObjectPosition(self.client_id, self.omniBot, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[0])
        position.append(tmp[1])

        res, tmp = vrep.simxGetObjectOrientation(self.client_id, self.omniBot, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[2]) # en radian

        return position
    
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

    def set_motor_velocity(self, control):
        """Set a target velocity on the omniBot motors, multiplied by the gain
        defined in self.gain

        Args:
            control(list): the control [left_motor, right_motor]
        """
        vrep.simxSetJointTargetVelocity(self.client_id, self.front_left_motor, self.gain*control[1], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetJointTargetVelocity(self.client_id, self.rear_right_motor, self.gain*control[0], vrep.simx_opmode_oneshot_wait)
