#!/usr/bin/env python

#Algorithm as defined in https://ieeexplore.ieee.org/document/9515484


import math
import numpy as np
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
import shapely
from shapely.geometry import LineString, Point

def distance_function(person1, person2, landmarks):

    k = 0.4

    person1_pose = np.array([person1[0]/100,person1[1]/100])
    person1_orient = np.array([math.cos(person1[2]),math.sin(person1[2])])
    person2_pose = np.array([person2[0]/100,person2[1]/100])
    person2_orient = np.array([math.cos(person2[2]),math.sin(person2[2])])
    
    vectnorm = np.linalg.norm(person1_pose - person2_pose)
    
    alphai = np.dot(person1_orient,(person2_pose - person1_pose)/vectnorm)
    alphaj = np.dot(person2_orient,(person1_pose - person2_pose)/vectnorm)
    
    orientation_factor = 1 - k*(alphai+alphaj)

    if landmarks[0] == landmarks[1] and landmarks[0] != 0 and landmarks[1] != 0:
        landmark_factor = 0.8
    else:
        landmark_factor = 1

    d = vectnorm*orientation_factor*landmark_factor


    return d
    
def landmark_detection(persons):

    lines = []
    intersections = []
    intersection_idx = []

    n_persons = len(persons)

    landmarks = np.zeros(n_persons)

    for i in range(0,n_persons):
        A = (persons[i][0]/100+0.3*math.cos(persons[i][2]),persons[i][1]/100+0.3*math.sin(persons[i][2]))
        B = (persons[i][0]/100+3*math.cos(persons[i][2]),persons[i][1]/100+3*math.sin(persons[i][2]))
        lines.append(LineString([A, B]))

    for i in range(0,n_persons):
        for j in range(i, n_persons):
            if lines[i].intersection(lines[j]) and i != j:
                
                intersect = lines[i].intersection(lines[j])
                intersections.append((intersect.x,intersect.y))
                intersection_idx.append((i,j))

    if len(intersections) > 1:

        Y = pdist(intersections)

        link_matrix = single(Y)

        clusters = fcluster(link_matrix, 0.5, criterion='distance')

        for i in range(1,len(clusters)+1):

            count = np.count_nonzero(clusters == i)

            if count > 2:
                for j in range(0,len(clusters)):
                    if clusters[j] == i:
                        landmarks[intersection_idx[j][0]] = i
                        landmarks[intersection_idx[j][1]] = i
            elif count == 0:
                break

        print(clusters)
    print(landmarks)

    return landmarks

def hierarchical_clustering(persons):

    n_persons = len(persons)
    group = []
    groups = []
    
    landmarks = landmark_detection(persons)

    dist_matrix = np.zeros((n_persons,n_persons))
    
    for i in range(0,n_persons):
        for j in range(i, n_persons):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                aux_dist = distance_function(persons[i],persons[j], (landmarks[i],landmarks[j]))
                dist_matrix[i][j] = aux_dist
                dist_matrix[j][i] = aux_dist

    link_matrix = single(dist_matrix)

    clusters = fcluster(link_matrix, 3, criterion='distance')


    for i in range(1, n_persons+1):
        for j in range(1, n_persons+1):
            if clusters[j-1] == i:
                group.append(persons[j-1])
        if group:
            groups.append(group)
            group = []
        else:
            break

    #print(groups)


    return groups
