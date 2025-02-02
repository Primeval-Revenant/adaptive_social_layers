#!/usr/bin/env python

import rospy
from group_msgs.msg import People, Person, Groups
from human_awareness_msgs.msg import PersonTracker, TrackedPersonsList
from geometry_msgs.msg import Pose, PoseArray, PointStamped
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

STRIDE = 65 # in cm

#Optimally, the width of the robot or a slightly larger value. Determines minimum size of approach zone.
MIN_DIST_SPACE = 0.8
OPEN_SPACE = 0.8

#Value which determines safety zone between person and approach zone
HUMAN_SIDE_FACTOR = 0.375

#Preset value for individual adaptation
INDIV_LOWER_VALUE = 0.3

#Variables that moderate the effect of velocity on the personal and group spaces
VEL_ADAPT_FACTOR = 1.5
GROUP_VEL_ADAPT_FACTOR = 1.5

#Variables that limit the maximum value of the velocity adaptation
ADAPT_LIMIT = 1
GROUP_ADAPT_LIMIT = 1

DISTANCE_ADAPT = 6 #Moderates velocity adaptation upon reaching a distance threshold

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
    """Euclidean distance between two points in 2D. Probably better substituted by just using numpy's norm"""
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

#https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
def point_on_line(a, b, p):
    """Project a point on a line, given 3 points, 2 of them forming said line"""
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result

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
        rospy.Subscriber("/human_trackers",TrackedPersonsList,self.callback,queue_size=1) #Subscribe to the human tracker
        rospy.Subscriber("/approach_target",PointStamped,self.callbackapproach,queue_size=1) #Subscribe to the approach target from the approach pose estimator
        rospy.Subscriber("/clicked_point",PointStamped, self.callbackPoint, queue_size=1) #Receive initial approach target
        self.loop_rate = rospy.Rate(rospy.get_param('~loop_rate', 10.0))
        self.pose_received = False
        self.target_received = False

        self.data = None
        self.approach_target = None
        self.pub = rospy.Publisher('/people', People, queue_size=1)
        self.pubg = rospy.Publisher('/groups', Groups, queue_size=1)

        self.pubd = rospy.Publisher('/people_detections', PoseArray, queue_size=1)

    def callback(self,data):
        """
        """
        
        self.data = data
        self.pose_received = True

    def callbackPoint(self,data):
        
        self.approach_target = data
        self.target_received = True

    def callbackapproach(self,data):
        """
        """
        
        self.approach_target = data
        self.target_received = True
        

    def publish(self):
        """
        """

        tfBuffer = tf.Buffer()

        listener = tf.TransformListener(tfBuffer)

        rate = rospy.Rate(10.0)
        aux_count_vel = 0
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
                self.pubd.publish(ap_points)
            elif data is not None:
                for poseinfo in data.personList:
                    
                    #Extract data about the existent people
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

                    # Pose transformation from odom frame to map frame
                    (px, py) = rotate(pose.position.x, pose.position.y, t_yaw)
                    pose_x = px + tx
                    pose_y = py + ty
                    pose_yaw = yaw + t_yaw


                    pose_person = (pose_x  * 100, pose_y * 100,  pose_yaw, poseinfo.velocity.linear.x,poseinfo.velocity.linear.y)
                    persons.append(pose_person)

                self.pubd.publish(ap_points) # Pose Array of individuals publisher
  
            #Run a clustering algorithm for group detection
            if persons:
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

                centers = []

                min_dist = 1000
                min_idx = -1

                #Calculate which group is closest to the approach target provided by the approaching algorithm. This allows for continuous tracking of moving groups
                for idx,group in enumerate(groups):
                    centers.append(calc_o_space(group))
                    if self.approach_target:
                        aux_dist = euclidean_distance(self.approach_target.point.x, self.approach_target.point.y, centers[idx][0]/100, centers[idx][1]/100)
                        if aux_dist < min_dist:
                            min_dist = aux_dist
                            min_idx = idx
                
                for idx,group in enumerate(groups):
                    aux_p = People()
                    aux_p.header.frame_id = "/map"
                    aux_p.header.stamp = rospy.Time.now()

                    sx = (float(pparams[idx][0])/100) # cm to m
                    sy = float(pparams[idx][1])/100 # cm to m
                    gvarx = float(gparams[idx][0]) / 100  # cm to m
                    gvary = float(gparams[idx][1]) / 100  # cm to m

                    center = centers[idx]

                    group = np.asarray(group, dtype=np.longdouble).tolist()
                    group.sort(key=lambda c: math.atan2(c[0]-center[0], c[1]-center[1])) #Sort group counterclockwise

                    #Variables used to calculate group velocity
                    sum_x_vel = 0
                    sum_y_vel = 0
                    sum_vel = 0

                    for i in range(len(group)):
                        p1 = Person()
                        p1.position.x = group[i][0] / 100 # cm to m
                        p1.position.y = group[i][1] / 100 # cm to m
                        p1.orientation = group[i][2]

                        vel_magnitude = np.linalg.norm([group[i][3],group[i][4]])
                        p1.velocity.linear.x = group[i][3]
                        p1.velocity.linear.y = group[i][4]

                        #Prepare variables for group velocity calculations
                        sum_x_vel += group[i][3]
                        sum_y_vel += group[i][4]
                        sum_vel += vel_magnitude

                        #Check distance between robot and person to check if it must moderate velocity adaptation
                        dist_pose = euclidean_distance(tx,ty,p1.position.x,p1.position.y)
                        if dist_pose > DISTANCE_ADAPT:
                            dist_modifier = 1
                        else:
                            dist_modifier = min(1,(dist_pose/DISTANCE_ADAPT)*2)

                        #Check if group or individual and if it is the chosen group to approach
                        if (len(group) != 1 or min_idx != idx):
                            p1.sx = min(sx*(1+dist_modifier*VEL_ADAPT_FACTOR*vel_magnitude),sx+ADAPT_LIMIT)
                        else:
                            lower_value = max(0.45,sx-INDIV_LOWER_VALUE)
                            p1.sx = min(lower_value*(1+dist_modifier*VEL_ADAPT_FACTOR*vel_magnitude),lower_value+ADAPT_LIMIT,sx+ADAPT_LIMIT,sx*(1+dist_modifier*VEL_ADAPT_FACTOR*vel_magnitude))

                        dist1 = 0
                        dist2 = 0

                        angle_dif = 0

                        #Check if it is the chosen group to approach and try to adapt the model if it is
                        if len(group) != 1 and min_idx == idx:
                            if len(group) == 2:
                                angle_dif = group[0][2] - group [1][2]
                                if angle_dif > math.pi:
                                    angle_dif -= 2*math.pi
                                elif angle_dif <= -math.pi:
                                    angle_dif += 2*math.pi

                                if i == 1:
                                    angle_dif = -angle_dif

                            #Calculate points HUMAN_SIDE_FACTOR of distance to the left and right of the person
                            aux_left = np.asarray((p1.position.x+HUMAN_SIDE_FACTOR*math.cos(p1.orientation+(math.pi/2)),p1.position.y+HUMAN_SIDE_FACTOR*math.sin(p1.orientation+(math.pi/2))))
                            aux_right = np.asarray((p1.position.x+HUMAN_SIDE_FACTOR*math.cos(p1.orientation-(math.pi/2)),p1.position.y+HUMAN_SIDE_FACTOR*math.sin(p1.orientation-(math.pi/2))))

                            #Calculate the equivalent point on the person to the left
                            if i != len(group)-1:
                                aux_left_adjacent = np.asarray(((group[i+1][0]/100)+HUMAN_SIDE_FACTOR*math.cos(group[i+1][2]-(math.pi/2)),(group[i+1][1]/100)+HUMAN_SIDE_FACTOR*math.sin(group[i+1][2]-(math.pi/2))))
                            else:
                                aux_left_adjacent = np.asarray(((group[0][0]/100)+HUMAN_SIDE_FACTOR*math.cos(group[0][2]-(math.pi/2)),(group[0][1]/100)+HUMAN_SIDE_FACTOR*math.sin(group[0][2]-(math.pi/2))))

                            #Distance between the points
                            dist1 = np.linalg.norm(aux_left-aux_left_adjacent)

                            #Calculate the equivalent point on the person to the right
                            if i != 0:
                                aux_right_adjacent = np.asarray(((group[i-1][0]/100)+HUMAN_SIDE_FACTOR*math.cos(group[i-1][2]+(math.pi/2)),(group[i-1][1]/100)+HUMAN_SIDE_FACTOR*math.sin(group[i-1][2]+(math.pi/2))))
                            else:
                                aux_right_adjacent = np.asarray(((group[len(group)-1][0]/100)+HUMAN_SIDE_FACTOR*math.cos(group[len(group)-1][2]+(math.pi/2)),(group[len(group)-1][1]/100)+HUMAN_SIDE_FACTOR*math.sin(group[len(group)-1][2]+(math.pi/2))))

                            #Distance between the points
                            dist2 = np.linalg.norm(aux_right-aux_right_adjacent)
                            
                            position_aux = np.asarray((p1.position.x,p1.position.y))

                            #Determine whether to adapt the left side
                            if dist1 > MIN_DIST_SPACE and (len(group) != 2 or angle_dif > 0):
                                aux_vector = (aux_left_adjacent-aux_left)/dist1
                                aux_point = aux_left+((dist1-OPEN_SPACE)/2)*aux_vector 
                                projected = point_on_line(aux_left,position_aux,aux_point)
                                dist_aux = np.linalg.norm(position_aux-projected)
                                p1.sy = min(dist_aux,sy)
                            else:
                                p1.sy = sy
                            
                            #Determine whether to adapt the right side
                            if dist2 > MIN_DIST_SPACE and (len(group) != 2 or angle_dif < 0):
                                aux_vector = (aux_right_adjacent-aux_right)/dist2
                                aux_point = aux_right+((dist2-OPEN_SPACE)/2)*aux_vector
                                projected = point_on_line(aux_right,position_aux,aux_point)
                                dist_aux = np.linalg.norm(position_aux-projected)
                                p1.sy_right = min(dist_aux,sy)
                            else:
                                p1.sy_right = sy

                        else:
                            p1.sy = sy
                            p1.sy_right = sy

                        p1.sx_back = sx / BACK_FACTOR
                        p1.ospace = False
                        p.people.append(p1)

                        
                        aux_p.people.append(p1)
                  
                    
                    # Only represent o space for  +2 individuals
                    if len(group) > 1:
                        p1 = Person()
                        p1.position.x = center[0] / 100 # cm to m
                        p1.position.y = center[1] / 100 # cm to m
                        p1.orientation = math.atan2(sum_y_vel,sum_x_vel) #orientation of group is orientation of the average velocity of all members
                        p1.velocity.linear.x = math.cos(p1.orientation)*(sum_vel/len(group))
                        p1.velocity.linear.y = math.sin(p1.orientation)*(sum_vel/len(group))

                        #Check distance between robot and group to check if it must moderate velocity adaptation
                        dist_pose = euclidean_distance(tx,ty,p1.position.x,p1.position.y)
                        if dist_pose > DISTANCE_ADAPT:
                            dist_modifier = 1
                        else:
                            dist_modifier = min(1,(dist_pose/DISTANCE_ADAPT)*2)

                        p1.sx = min(gvarx*(1 + dist_modifier*GROUP_VEL_ADAPT_FACTOR*(sum_vel/len(group))),gvarx+GROUP_ADAPT_LIMIT)
                        p1.sx_back = gvarx
                        p1.sy = gvary
                        p1.ospace = True
                        p.people.append(p1)

                        aux_p.people.append(p1)


                    aux_p.id = str(idx)
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