#!/usr/bin/env python



import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from group_msgs.msg import People, Person, Groups
from geometry_msgs.msg import Pose, PoseArray
from algorithm import SpaceModeling
from obstacles import adapt_parameters
import tf as convert
import tf2_ros as tf

import math
import copy
import actionlib
import numpy as np

import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'/home/ricarte/catkin_ws/src/adaptive_social_layers/scripts', nargout=0)

STRIDE = 65 # in cm
MDL = 8000

# Relation between personal frontal space and back space
BACK_FACTOR = 1.3

# Robot radius
ROBOT_DIM = 100 # in cm

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

def rotate(px, py, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py

    return qx, qy

def get_index(x, y, width):
    """ """
    
    return (y * width) + x

class PeoplePublisher():
    """
    """
    def __init__(self):
        """
        """
        rospy.init_node('PeoplePublisher', anonymous=True)
        
        rospy.Subscriber("/faces",PoseArray,self.callback,queue_size=1)
                # https://answers.ros.org/question/207620/global-map-update/
        # We need to subscribe both costmap and costmap update topic
        rospy.Subscriber("/map",OccupancyGrid , self.callbackCm, queue_size=1)
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10.0))

        self.map_received = False
        self.pose_received = False

        self.data = None
        self.pub = rospy.Publisher('/people', People, queue_size=10)
        self.pubg = rospy.Publisher('/groups', Groups, queue_size=10)


    def callback(self,data):
        """
        """
        
        self.data = data
        self.pose_received = True
    

    def callbackCm(self, data):
        """ Costmap Callback. """

        self.map = data
        self.map_grid = list(data.data)
        self.map_received = True
            
    def publish(self):
        """
        """
        

        tfBuffer = tf.Buffer()

        listener = tf.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                transf = tfBuffer.lookup_transform('map', 'base_footprint', rospy.Time())
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue

            data = self.data
            groups = []
            group = []

            persons = []

            tx = transf.transform.translation.x
            ty = transf.transform.translation.y
            quatern = (transf.transform.rotation.x, transf.transform.rotation.y, transf.transform.rotation.z, transf.transform.rotation.w)
            (_, _, t_yaw) = convert.transformations.euler_from_quaternion(quatern)

            if not data.poses:
                groups = []
            else:
                for pose in data.poses:

                    #rospy.loginfo("Person Detected")
                    
                    quartenion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    (_, _, yaw) = convert.transformations.euler_from_quaternion(quartenion)

                    
                    # Pose transformation from base footprint frame to map frame
                    (px, py) = rotate(pose.position.x, pose.position.y, t_yaw)
                    pose_x = px + tx
                    pose_y = py + ty
                    pose_yaw = yaw + t_yaw


                    pose_person = (pose_x  * 100, pose_y * 100,  pose_yaw)
                    persons.append(pose_person)

            # Run GCFF gcff.m Matlab function      
            if persons:
                groups = eng.gcff(MDL,STRIDE, matlab.double(persons))
    
            if groups:
                app = SpaceModeling(groups) # Space modeling works in cm
                pparams,gparams = app.solve()

                ####
                # Obstacles works in cm -> Convert to meters
                ox = self.map.info.origin.position.x * 100
                oy = self.map.info.origin.position.y * 100
                origin = [ox, oy]
                resolution = self.map.info.resolution * 100
                width = self.map.info.width 
                height = self.map.info.height 
                map = self.map.data

                pparams_adapt, gparams_adapt = adapt_parameters(groups, pparams, gparams, resolution, map, origin, width, ROBOT_DIM)
                

                p = People()
                p.header.frame_id = "/map"
                p.header.stamp = rospy.Time.now()

                g = Groups()
                g.header.frame_id = "/map"
                g.header.stamp = rospy.Time.now()
                
                for idx,group in enumerate(groups):
                    aux_p = People()
                    aux_p.header.frame_id = "/map"
                    aux_p.header.stamp = rospy.Time.now()

                    
                    gvarx = float(gparams_adapt[idx][0]) / 100.0  # cm to m
                    gvary = float(gparams_adapt[idx][1]) / 100.0  # cm to m
                    
        
                    
                    ############## FIXED
                    # sx = 0.9
                    # sy = 0.9
                    #########################
                    for pidx, person in enumerate(group):

                        p1 = Person()
                        p1.position.x = person[0] / 100.0 # cm to m
                        p1.position.y = person[1] / 100.0 # cm to m
                        p1.orientation = person[2]

                        sx = pparams_adapt[idx][pidx]["sx"]/ 100.0
                        sy =  pparams_adapt[idx][pidx]["sy"] / 100.0

                        sx_back = pparams_adapt[idx][pidx]["sx_back"] / 100.0
                        
                        p1.sx = sx 
                        p1.sy = sy 
                        p1.sx_back = sx_back 
                        
                        p1.ospace = False
                        p.people.append(p1)

                        
                        aux_p.people.append(p1)
                    
                    # Only represent o space for  +2 individuals
                    if len(group) > 1:
                        p1 = Person()
                        center = calc_o_space(group)
                        p1.position.x = center[0] / 100.0 # cm to m
                        p1.position.y = center[1] / 100.0 # cm to m
                        p1.orientation = math.pi

                        
                        p1.sx = gvarx
                        p1.sy = gvary

                        
                        p1.ospace = True
                        p.people.append(p1)

                        aux_p.people.append(p1)

                    g.groups.append(aux_p)

                self.pub.publish(p)
                
                self.pubg.publish(g)

        else:
            p = People()
            p.header.frame_id = "/map"
            p.header.stamp = rospy.Time.now()
            self.pub.publish(p)

            g = Groups()
            g.header.frame_id = "/map"
            g.header.stamp = rospy.Time.now()
            self.pubg.publish(g)

    def run_behavior(self):
        while not rospy.is_shutdown():
            if self.pose_received:
                self.pose_received = False

                if self.map_received:
                    #self.map_received = False

                    self.publish()
                    

if __name__ == '__main__':
    people_publisher = PeoplePublisher()
    people_publisher.publish()
    eng.quit()