#!/usr/bin/env python



import rospy
from group_msgs.msg import People, Person, Groups
from human_awareness_msgs.msg import PersonTracker, TrackedPersonsList
from geometry_msgs.msg import Pose, PoseArray
import tf as convert
import tf2_ros as tf
import tf_conversions as tfc
import math
from algorithm import SpaceModeling
import copy
from visualization_msgs.msg import Marker
import numpy as np
from clustering_algorithm import hierarchical_clustering

import actionlib

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.cd(r'/home/ricarte/catkin_ws/src/adaptive_social_layers/scripts', nargout=0)


STRIDE = 65 # in cm
MDL = 8000

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

def euclidean_distance(x1, y1, x2, y2):
    """Euclidean distance between two points in 2D."""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def rotate(px, py, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py

    return qx, qy

class PeoplePublisher():
    """
    """
    def __init__(self):
        """
        """
        rospy.init_node('PeoplePublisher', anonymous=True)
        
        #rospy.Subscriber("/faces",PoseArray,self.callback,queue_size=1)
        rospy.Subscriber("/human_trackers",TrackedPersonsList,self.callback,queue_size=1)
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10.0))
        self.pose_received = False

        self.data = None
        self.pub = rospy.Publisher('/people', People, queue_size=1)
        self.pubg = rospy.Publisher('/groups', Groups, queue_size=1)

        self.pubd = rospy.Publisher('/people_detections', PoseArray, queue_size=1)

    def callback(self,data):
        """
        """
        
        self.data = data
        self.pose_received = True
        

    def publish(self):
        """
        """

        tfBuffer = tf.Buffer()

        listener = tf.TransformListener(tfBuffer)

        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:

                transf = tfBuffer.lookup_transform('map', 'odom', rospy.Time())
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
            

            ap_points = PoseArray()
            ap_points.header.frame_id = "/odom"
            ap_points.header.stamp = rospy.Time.now()

            if (data is not None) and (not data.personList):
                groups = []
            elif data is not None:
                for poseinfo in data.personList:

                    #rospy.loginfo("Person Detected")

                    pose = poseinfo.body_pose

                    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                    ###################### Pose Array Marker of the individuals
                    ap_pose = Pose()
                    ap_pose.position.x = pose.position.x
                    ap_pose.position.y = pose.position.y
                    ap_pose.position.z = 0.1

                    ap_pose.orientation.x = quaternion[0]
                    ap_pose.orientation.y = quaternion[1]
                    ap_pose.orientation.z = quaternion[2]
                    ap_pose.orientation.w = quaternion[3]
                    
                    ap_points.poses.append(ap_pose)
                    #########################
                    (_, _, yaw) = convert.transformations.euler_from_quaternion(quaternion)

                    # Pose transformation from base footprint frame to map frame
                    (px, py) = rotate(pose.position.x, pose.position.y, t_yaw)
                    pose_x = px + tx
                    pose_y = py + ty
                    pose_yaw = yaw + t_yaw


                    pose_person = (pose_x  * 100, pose_y * 100,  pose_yaw)
                    persons.append(pose_person)

                self.pubd.publish(ap_points) # Pose Array of individuals publisher

            # Run GCFF gcff.m Matlab function     
            if persons:
                #groups = eng.gcff(MDL,STRIDE, matlab.double(persons))
                groups = hierarchical_clustering(persons)
    
            if groups:
                app = SpaceModeling(groups) # Space modeling works in cm
                pparams,gparams = app.solve()

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

                    sx = (float(pparams[idx][0])/100) # cm to m
                    sy = float(pparams[idx][1])/100 # cm to m
                    gvarx = float(gparams[idx][0]) / 100  # cm to m
                    gvary = float(gparams[idx][1]) / 100  # cm to m

                    center = calc_o_space(group)
                    group = np.asarray(group, dtype=np.longdouble).tolist()
                    group.sort(key=lambda c: math.atan2(c[0]-center[0], c[1]-center[1]))

                    ############## FIXED
                    #sx = 0.9
                    #sy = 0.9
                    #########################
                    for i in range(len(group)):
                        p1 = Person()
                        p1.position.x = group[i][0] / 100 # cm to m
                        p1.position.y = group[i][1] / 100 # cm to m
                        p1.orientation = group[i][2]
                        
                        p1.sx = sx
                        p1.sy = sy
                        p1.sy_right = sy

                        p1.sx_back = p1.sx / BACK_FACTOR
                        p1.ospace = False
                        p.people.append(p1)

                        
                        aux_p.people.append(p1)
                  
                    
                    # Only represent o space for  +2 individuals
                    if len(group) > 1:
                        p1 = Person()
                        p1.position.x = center[0] / 100 # cm to m
                        p1.position.y = center[1] / 100 # cm to m
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
            rate.sleep()

if __name__ == '__main__':
    people_publisher = PeoplePublisher()
    people_publisher.publish()
    eng.quit()