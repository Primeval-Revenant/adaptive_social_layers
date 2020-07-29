#!/usr/bin/env python

import rospy
from group_msgs.msg import People, Person
from geometry_msgs.msg import Pose, PoseArray
import tf
import math
from algorithm import SpaceModeling
import copy

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


STRIDE = 0.65 # in m

# Relation between personal frontal space and back space
BACK_FACTOR = 1.3

def calc_o_space(persons):
    """Calculates the o-space center of the group given group members pose"""
    c_x = 0
    c_y = 0
    
# Group size
    g_size = len(persons)
    


    for person in persons:
        c_x += person[0] + math.cos(person[2]) * STRIDE
        c_y += person[1] + math.sin(person[2]) * STRIDE

    center = [c_x / g_size, c_y / g_size]


    return center

class PeoplePublisher():
    def __init__(self):
        rospy.init_node('talker', anonymous=True)
        
        rospy.Subscriber("/faces",PoseArray,self.callback,queue_size=1)
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10.0))
        self.pose_received = False

        self.data = None
        self.pub = rospy.Publisher('/people', People, queue_size=1)
        

    def callback(self,data):
        
        self.data = data
        self.pose_received = True
        

    def publish(self):
        
        data = self.data
        group = []
        if not data.poses:
            group = []
        else:
            for pose in data.poses:

                rospy.loginfo("Person Detected")
                
                quartenion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quartenion)

                pose_person = [pose.position.x, pose.position.y,yaw]
                


                group.append(pose_person)
       
        if group:
            aux_group = copy.deepcopy(group)
            groups = [aux_group]
            for gp in groups:
                for p in gp:
                    p[0] = p[0] * 100 #algorithm uses cm
                    p[1] = p[1] * 100 # m to cm

            

            app = SpaceModeling(groups)
            pparams,gparams= app.solve()


            

            sx = (float(pparams[0][0])/100)/BACK_FACTOR # cm to m
            sy = float(pparams[0][1])/100 # cm to m
            gvar = float(gparams[0]) / 100  # cm to m

            
            p = People()
            p.header.frame_id = "/base_footprint"
            p.header.stamp = rospy.Time.now()

  
            for person in group:

                p1 = Person()
                p1.position.x = person[0]
                p1.position.y = person[1]
                p1.position.z = person[2]
                p1.orientation = person[2]
                p1.sx = sx
                p1.sy = sy
                p.people.append(p1)

            
            p1 = Person()
            center = calc_o_space(group)
            p1.position.x = center[0]
            p1.position.y = center[1]
            p1.orientation = math.pi
            p1.sx = gvar
            p1.sy = gvar
            p.people.append(p1)
            self.pub.publish(p)


 

        else:
            p = People()
            p.header.frame_id = "/base_footprint"
            p.header.stamp = rospy.Time.now()
            self.pub.publish(p)

    def run_behavior(self):
        while not rospy.is_shutdown():
            if self.pose_received:
                
                self.pose_received = False
                self.publish()

if __name__ == '__main__':

 
    people_publisher = PeoplePublisher()
    people_publisher.run_behavior()
