#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 00:48:54 2018

@author: amritgos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mode
import collections
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

#Quick Sort
def quicksort(x):
    if len(x) == 1 or len(x) == 0:
        return x
    else:
        pivot = x[0]
        i = 0
        for j in range(len(x)-1):
            if x[j+1] < pivot:
                x[j+1],x[i+1] = x[i+1], x[j+1]
                i += 1
        x[0],x[i] = x[i],x[0]
        first_part = quicksort(x[:i])
        second_part = quicksort(x[i+1:])
        first_part.append(x[i])
        return first_part + second_part
    
#Distance
def eval_distance(data):
    N = len(data)
    distance = np.zeros((N,N))
    for i in range (N):
        distance[i] = np.linalg.norm(data[i] - data, axis=1)
    return distance


#Density
def Local_density(data,epsilon):
    density = np.zeros(len(data))
    for i in range (len(data)):
        density[i] = np.sum(np.exp((-1)*((np.linalg.norm(data[i] - data, axis=1))**2)/(2*(epsilon**2))))
    return density


#Random Population
def Random_pop(data,size,n_clusters):
    X = np.random.randint(n_clusters+1, size=(size,len(data)))
    return X


def cluster_size(solution):
    cluster_list = []
    for i in range(len(solution)):
        if (solution[i] not in cluster_list):
            cluster_list.append(solution[i])
    clusters = np.zeros((len(cluster_list),3))
    for j in range(len(cluster_list)):
        val = cluster_list[j]
        count = 0
        for i in range(len(solution)):
            if solution[i] == val :
                count += 1
        clusters[j][0] = val
        clusters[j][1] = count
    s = np.sum(clusters , axis = 0)
    s = s[1]
    clusters[0][2] = clusters[0][1]/s 
    for i in range(1,len(clusters)):
        clusters[i][2] = clusters[i-1][2] + clusters[i][1]/s
    return clusters

#Interaction Graph
def Interaction_graph(k,data,epsilon):
    N = len(data)
    dist = eval_distance(data)
    dens = Local_density(data,epsilon)
    G = np.zeros((N,N))
    
    #Max-density point
    min_dist = 1000
    max_density_index = np.argmax(dens, axis = 0)
    for j in range(N):
        if (dist[max_density_index][j] <= min_dist ):
            min_dist = dist[max_density_index][j]
            min_max_dist_index = j
    G[max_density_index][min_max_dist_index] = 1  
    
    for i in range(N):
        G[i][i] = 1
        if i != max_density_index :
            if i != min_max_dist_index:
                min_dist = 1000
                for j in range(N):
                    if dens[j] > dens[i]:
                        if (dist[i][j] <= min_dist ):
                            min_dist = dist[i][j]
                            min_dist_index = j
                G[i][min_dist_index] = 1
        l_min = []
        while(np.sum(G[i]) != k+1):
            min_dist = 1000
            for j in range(N):
                if j not in l_min:
                    if (dist[i][j] <= min_dist ):
                        min_dist = dist[i][j]
                        min_dist_index = j
            G[i][min_dist_index] = 1
            l_min.append(min_dist_index)
    Gep = np.zeros((N,k+1))
    for i in range(N):
        count = 0
        for j in range(N):
            if(G[i][j] == 1):
                Gep[i][count] = j
                count += 1
    return Gep


#Threshold
def init_threshold(data,epsilon,dens,dist):
    N = len(data)
    
    #Calculate Cp
    mp = np.mean(dens)
    stdp = np.std(dens)
    if (mp - stdp > 0) :
        cp = mp - stdp
    else :
        cp = mp/2
    
    #Calculating C1,C2,C3
    for i in range(N):
        m = np.mean(dist)
        std = np.std(dist)
        c1 = m
        c2 = m + std
        c3 = m + 2*std
        
    return c1,c2,c3,cp


def sub_func(k,dens,dist,Gep,solution,c1,c2,c3,cp,i):
    fi = 0
    i = int(i)
    for j in range(k+1):
        if(Gep[i][j] != i):
            index = int(Gep[i][j])
            #Noise
            if (solution[i] == 0) :
                i = int(i)
                if (dist[i][index] > c2 and dens[i] <= cp):
                    alpha = 0
                else :
                    alpha = dens[index]
            #Same Cluster
            i = int(i)
            index = int(index)
            if (solution[i] == solution[index]):
                if dist[i][index] < c1 :
                    alpha = 0
                elif dist[i][index] >= c1 and dist[i][index] <= c3 :
                    alpha = ((dist[i][index] - c1)/(c3-c1))*dens[index]
                else :
                    alpha = dens[index]
            #Otherwise
            else :
                if dist[i][index] < c1 :
                    alpha = dens[index]
                elif dist[i][index] >= c1 and dist[i][index] <= c3 :
                    alpha = ((c3 - dist[i][index])/(c3-c1))*dens[index]
                else :
                    alpha = 0
            fi += alpha
    return fi
    
            
def func(data,solution,k,epsilon,Gep):
    N = len(data)
    f = np.zeros(N)
    dens = Local_density(data,epsilon)
    dist = eval_distance(data)
    
    c1,c2,c3,cp = init_threshold(data,epsilon,dens,dist)
    
    #Evaluate fi's
    for i in range(N):
        f[i] = sub_func(k,dens,dist,Gep,solution,c1,c2,c3,cp,i)
    
    f = np.sum(f)
    
    return f


def eval_delta(data,solution,Gep,k,epsilon,obs_index,v):
    solution_new = solution
    solution_new[obs_index] = v
    dens = Local_density(data,epsilon)
    dist = eval_distance(data)
    c1,c2,c3,cp = init_threshold(data,epsilon,dens,dist)
    delta = []
    for i in range(len(data)):
        for j in range(k+1):
            if Gep[i][j] ==  obs_index :
                f_old = sub_func(k,dens,dist,Gep,solution,c1,c2,c3,cp,i)
                f_new = sub_func(k,dens,dist,Gep,solution_new,c1,c2,c3,cp,i)
                d = f_new - f_old
                delta.append(d)
    delta = np.asarray(delta)
    delta_sum =np.sum(delta)
    return delta_sum


def local_search(data,solution,Gep,k,epsilon,num_iter):
    for counter in range(num_iter):
        i = random.randint(0,(len(data)-1))
        min_delta = 0
        count = 0
        for j in Gep[i] :
            i != j
            j = int(j)
            delta = eval_delta(data,solution,Gep,k,epsilon,i,solution[j])
            if delta <= min_delta :
                min_delta = delta
                arg_min_delta = j
                count += 1
        if count != 0 :
            solution[i] = solution[arg_min_delta]
    return solution
            
            
def Initialise_pop(data,size,n_clusters,Gep,k,epsilon):
    X = Random_pop(data,size,n_clusters)
    for i in range(len(X)):
        X[i] = local_search(data,X,Gep,k,epsilon)
    return X 


def best_selection(population,data,k,epsilon,Gep):
    fitness = np.zeros(len(population))
    for i in range(len(population)):
        fitness[i] = func(data,population[i],k,epsilon,Gep)
    min_fit = np.argmin(fitness)
    return min_fit

def best_selection2(population,data,k,epsilon,best_index,Gep):
    fitness = np.zeros(len(population))
    fitness[best_index] = 100000
    for i in range(len(population)):
        if i == best_index :
            continue
        fitness[i] = func(data,population[i],k,epsilon,Gep)
    min_fit = np.argmin(fitness)
    return min_fit


def mutation_NK(data,solution,Gep,k,epsilon):
    i = random.randint(0,(len(data)-1))
    min_delta = 10000
    for j in Gep[i] :
        i != j
        j = int(j)
        delta = eval_delta(data,solution,Gep,k,epsilon,i,solution[j])
        if delta <= min_delta :
            min_delta = delta
            arg_min_delta = j
    solution[i] = solution[arg_min_delta]
    return solution 


def renum(X1,X2):
    X2_m = X2
    clusters = []
    for i in range(len(X2)):
        if(X2[i] not in clusters):
            clusters.append(X2[i])
    num_clusters_2 = len(clusters)
    
    clusters = []
    for i in range(len(X1)):
        if(X2[i] not in clusters):
            clusters.append(X1[i])
    num_clusters_1 = len(clusters)
    
    num_clusters = num_clusters_2
    
    map_clusters = np.zeros((num_clusters,2))
        
    clusters = []
    count = 0
    for i in range(len(X2)):
        X1m = []
        if(X2[i] not in clusters):
            clusters.append(X2[i])
            for j in range(len(X2)):
                if (X2[j] == X2[i]):
                    X1m.append(X2[j])
            val = mode(X1m)
            map_clusters[count][0] = val
            map_clusters[count][1] = X2[i]
            count += 1
    for i in range(len(X2)):
        for j in range(len(map_clusters)):
            if X2[i] == map_clusters[j][1]:
                X2_m[i] = map_clusters[j][0]
    return X2_m 

def cluster_proto(solution,num_clusters,density,clusters_list):
    cluster_prototype = np.zeros((num_clusters,2))
    for j in range(num_clusters):
        max_density = 0
        cluster = int(clusters_list[j])
        flag = 0
        for i in range(len(solution)):
            if solution[i] == cluster :
                if density[i] >= max_density :
                    max_density = density[i]
                    index = i
                    flag = 1
        if flag == 1:
            cluster_prototype[j][0] = cluster
            cluster_prototype[j][1] = index
        else :
            print("Error")
    return cluster_prototype


def mutation_merge(data,solution,density):
    clusters_list = []
    for i in range(len(solution)):
        if(solution[i] not in clusters_list):
            clusters_list.append(solution[i])
            
    num_clusters = len(clusters_list)
    
    cluster_prototype = cluster_proto(solution,num_clusters,density,clusters_list)
    
    #selecting random cluster
    r = random.randint(0,len(clusters_list)-1)
    cluster = int(clusters_list[r])
    for i in range(len(cluster_prototype)):
        if int(cluster_prototype[i][0]) == cluster :
            cluster_prototype_index = int(cluster_prototype[i][1])
            break
        
    #selecting second cluster
    d = np.full((num_clusters-1,3),100,dtype = float)
    # 1st column denotes cluster label
    # 2nd denotes d_inv
    # 3rd denotes cumulative prob 
    c = 0
    for i in range(num_clusters):
        cluster_new = int(clusters_list[i])
        
        if int(cluster_new) == int(cluster) :
            continue
        d[c][0] = int(cluster_new)
        d[c][0] = int(d[c][0])
        y1 = data[cluster_prototype_index]
        
        for j in range(len(cluster_prototype)):
            if cluster_prototype[j][0] == cluster_new :
                cluster_prototype_index_new = cluster_prototype[j][1]
                break
        y2 = data[int(cluster_prototype_index_new)]
        dy = y2-y1
        dy = np.asarray(dy)
        d[c][1] = float(np.linalg.norm(dy))
        d_inv = float(1/(d[c][1]))
        d[c][1] = float(d_inv)
        c += 1
    sum_d = np.sum(d,axis=0)
    s = sum_d[1] 
    if len(d) != 0 :
        return solution
    d[0][2] = d[0][1]/s
    for i in range(1,len(d)):
        d[i][2] = d[i-1][2] + (d[i][1]/s)
    
    r = random.random()
    
    for i in range(len(d)):
        if r < (d[0][2]) :
            index_cluster = int(d[0][0])
            break
        
        if r <= (d[i][2]) :
            if r > (d[i-1][2]) :
                index_cluster = int(d[i][0])
    
    for i in range(len(data)):
        if (solution[i] == cluster):
            solution[i] = index_cluster
    return solution
                

def mutation_split(data,sol,density):
    clusters_list = []
    solution = sol
    for i in range(len(solution)):
        if(solution[i] not in clusters_list):
            clusters_list.append(solution[i])

    clusters = cluster_size(solution)
    num_clusters = len(clusters_list)
    
    #Cluster to be split
    r = random.random()
    for i in range(len(clusters)):
        if r < clusters[0][2] :
            index_cluster = int(clusters[0][0])
            break
        
        if r < clusters[i][2] :
            if r >= clusters[i-1][2] :
                index_cluster = int(clusters[i][0])
                break
            
    cluster_prototype = cluster_proto(solution,num_clusters,density,clusters_list)
    
    #Finding 2 new centers
    for i in range(len(cluster_prototype)):
        if int(cluster_prototype[i][0]) == index_cluster:
            p1 = int(cluster_prototype[i][1])
            break
    
    max_density = 0
    count = 0
    for i in range(len(solution)):
        if int(solution[i]) == index_cluster :
            if density[i] >= max_density :
                if i != p1 :
                    max_density = density[i]
                    p2 = i
                    count += 1
    #new index
    max_index = 0
    for i in range(len(clusters)):
        if clusters[i][0] >= max_index:
            max_index = clusters[i][0]
    new_index = max_index + 1
        
        
    for i in range(len(solution)):
        if (int(solution[i]) == int(index_cluster)):
            if count != 0:
                d1 = np.linalg.norm(data[i] - data[p1])
                d2 = np.linalg.norm(data[i] - data[p2])
                
                if d1 >= d2 :
                    continue
                
                else :
                    solution[i] = new_index
            
    return solution



def bfs(graph, root): 
    visited, queue = set(), collections.deque([root])
    visited.add(root)
    while queue: 
        vertex = queue.popleft()
        for neighbour in graph[vertex]: 
            if neighbour not in visited: 
                visited.add(neighbour) 
                queue.append(neighbour)
    return visited

def Grec_dict(Gep):
    graph_dict = {}
    for i in range(len(Gep)):
        Gi = []
        for j in range(len(Gep[i])):
            index = Gep[i][j]  
            Gi.append(index)
        graph_dict[i] = Gi
    return graph_dict
    
    
def Partition_crossover(Gep,data,sol1,sol2,k,dens,dist):
    Gnew = Gep
    sol = sol1
    vertex = []
    for i in range(len(Gep)):
        vertex.append(i)
    for i in range(len(data)):
        if (sol1[i] == sol2[i]):
            vertex.remove(i)
            np.delete(Gnew, (i), axis=0)
    for i in range(len(Gnew)):
        for j in range(len(Gnew[i])):
            if Gnew[i][j] not in vertex :
                np.delete(Gnew[i],j)
    graph = Grec_dict(Gnew)
    visited = []
    for i in range(len(Gnew)):
        visit = bfs(graph,i)
        for j in visit :
            if j not in visited :
                visited.append(j)
    c1,c2,c3,cp = init_threshold(data,epsilon,dens,dist)
    q = len(visited)
    for j in range(q):
        index = int(visited[j])
        h1j = sub_func(k,dens,dist,Gep,sol1,c1,c2,c3,cp,index)
        h2j = sub_func(k,dens,dist,Gep,sol2,c1,c2,c3,cp,index)
        if h1j > h2j:
            sol1[index] = sol2[index]
    return sol


def NKCV2(data,size_pop,n_clusters,epsilon,k,num_iter,pc,n_ls):
    density = Local_density(data,epsilon)
    distance = eval_distance(data)
    Gep = Interaction_graph(k,data,epsilon)
    print("Graph Prepared")
    size = size_pop
    X = Random_pop(data,size,n_clusters)
    for i in range(size):
        X[i] = local_search(data,X[i],Gep,k,epsilon,n_ls)
    print("Initialised")
    t = 1
    Q = np.zeros((size_pop,len(data)))
    Best = np.zeros(len(data))
    fit = np.zeros(num_iter)
    min_fitness_sol = func(data,X[0],k,epsilon,Gep)
    while t <= num_iter:
        best_sol = best_selection(X,data,k,epsilon,Gep)
        Q[0] = X[best_sol]
        Best = Q[0]
        for i in range(1,size_pop):
            p1 = X[best_sol]
            r = random.random()
            if r < pc :
                p2_index = best_selection2(X,data,k,epsilon,best_sol,Gep)
                p2 = X[p2_index]
                p2_new = renum (p1,p2)
                #To be implemented
                Q[i] = Partition_crossover(Gep,data,p1,p2_new,k,density,distance)
                #Q[i] = fix_labels()
                #End
            else:
                r = random.random()
                if r < 0.6 :
                    Q[i] = mutation_NK(data,p1,Gep,k,epsilon)
                    
                elif r < 0.8 :
                    Q[i] = mutation_merge(data,p1,density)
                    
                else:
                    sol = mutation_split(data,p1,density)
                    Q[i] = sol
        if t%50 == 0:
            print("Iteration :",t)
            
        if t%100 == 0:
            Q_new = np.zeros((size_pop,len(data)))
            best_sol = best_selection(Q,data,k,epsilon,Gep)
            Q_new[0] = Q[best_sol]
            
            for i in range(1,size_pop):
                if i < 0.7*size_pop:
                    Q_new[i] = local_search(data,Q[i],Gep,k,epsilon,n_ls)
                else:
                    Q_new[i] = np.random.randint(0,n_clusters+1,size = (1,len(data)))
                    Q_new[i] = local_search(data,Q_new[i],Gep,k,epsilon,n_ls)
            Q = Q_new
            
        X = Q
        best_sol = best_selection(X,data,k,epsilon,Gep)
        if func(data,X[best_sol],k,epsilon,Gep) <= func(data,Best,k,epsilon,Gep):
            X[0] = X[best_sol]
        else:
            X[0] = Best
            
        fit[t-1] = func(data,X[best_sol],k,epsilon,Gep)
        
        best_sol = best_selection(X,data,k,epsilon,Gep)
        t += 1
        
    best_sol = best_selection(X,data,k,epsilon,Gep)
    return (X[best_sol])

def eval_distance_init(data):
    N = len(data)
    distance = np.zeros((N,N))
    for i in range (N):
        c1 = ((data[i][0],data[i][1]))
        for j in range(N):
            c2 = ((data[j][0],data[j][1]))
            d = (great_circle(c1,c2).km)
            distance[i][j] = d 
    return distance

def cluster_split(data,data_org,sol,density,index_cluster):
    clusters_list = []
    solution = sol
    for i in range(len(solution)):
        if(solution[i] not in clusters_list):
            clusters_list.append(solution[i])

    clusters = cluster_size(solution)
    num_clusters = len(clusters_list)

            
    cluster_prototype = cluster_proto(solution,num_clusters,density,clusters_list)

    #Finding 2 new centers
    for i in range(len(cluster_prototype)):
        if int(cluster_prototype[i][0]) == index_cluster:
            p1 = int(cluster_prototype[i][1])
            break
    c1 = ((data_org[p1][0],data_org[p1][1]))
    max_density = 0
    count = 0
    for i in range(len(solution)):
        if int(solution[i]) == index_cluster :
            if density[i] >= max_density :
                c2 = ((data_org[i][0],data_org[i][1]))
                d = (great_circle(c1,c2).km)
                if i != p1 :
                    if d >= 5:
                        max_density = density[i]
                        p2 = i
                        count += 1
    #new index
    max_index = 0
    for i in range(len(clusters)):
        if clusters[i][0] >= max_index:
            max_index = clusters[i][0]
    new_index = max_index + 1
        
        
    for i in range(len(solution)):
        if (int(solution[i]) == int(index_cluster)):
            if count != 0:
                d1 = np.linalg.norm(data[i] - data[p1])
                d2 = np.linalg.norm(data[i] - data[p2])
                
                if d1 >= d2 :
                    continue
                
                else :
                    solution[i] = new_index
            
    return solution

                          #Main     
#Parameters     
pi = 200
lamda = 200
#No.of tentative proc. centres
num_pc_val = 20                     # change 
#max dist. b/w village and pc allowed
vil_pc_maxdistance = 15
num_cw_init = 8


size_pop = 20
n_clusters = 15
epsilon = 2
k = 3
num_iter = 300
pc = 0.6
n_ls = 20


train = "Kapurthala.csv" 
df = pd.read_csv(train)
#print(df.head())
print("Villages")
data = []
for i in range(len(df)):
    d = []
    d.append(float(df["Latitude"][i]))
    d.append(float(df["Longitude"][i]))
    data.append(d)
data = np.asarray(data)


X = np.asarray(df["Latitude"])
Y = np.asarray(df["Longitude"])
plt.scatter(X,Y)
plt.show()

epsilon = 0.05

num_pc = num_pc_val

density = Local_density(data,epsilon)
density_copy = Local_density(data,epsilon)
density_list = []
#print(density)
i = 0
while i < num_pc:
    flag = 1
    max_density = 0
    for j in range(len(density_copy)):
        if density_copy[j] > max_density:
            max_density = density_copy[j]
            index = j
        c1 = ((data[index][0],data[index][1]))
    for j in density_list:
        c2 = ((data[j][0],data[j][1]))
        d = (great_circle(c1,c2).km)
        if d < 6 :
            flag = 0
            break
    if flag == 1:
        density_list.append(index)
        i += 1
    density_copy[index] = 0
print("Villages with Procurement Centres")
for j in range(len(data)):
    if j in density_list:
        plt.scatter(data[j][0],data[j][1],marker='*',c='b',s = 100)
    else:
        plt.scatter(data[j][0],data[j][1],s = 40,c='b')
plt.show()
distance = eval_distance_init(data)
data_new = np.zeros((len(data),num_pc))
for i in range(len(density_list)):
    for j in range(len(data_new)):
        index = int(density_list[i])
        data_new[j][i] = distance[j][index]
        
num_cw = num_cw_init
num_pc = num_pc_val
cw = np.zeros((num_cw,2))
cwx = np.random.random((num_cw,1))
cwy = np.random.random((num_cw,1))
for i in range(len(cw)):
    cw[i][0] = 31 + cwx[i]
    cw[i][1] = 75 + cwy[i]
               
#print(cw)
print("Central Warehouse")
for j in range(len(cw)):
    plt.scatter(cw[j][0],cw[j][1],s = 40,c = 'r')
plt.xlim(31,32)
plt.ylim(75,76)
plt.show()

fc_init = np.random.randint(2000000,4000000, size = (len(density_list)))

c1 = np.zeros((len(data),num_pc,3))
for i in range (len(data)):
    for j in range(num_pc):
        r = np.random.random()
        if r < 0.33:
            c1[i][j][0] = 1
        elif r < 0.67:
            c1[i][j][1] = 1
        else :
            c1[i][j][2] = 1
c1_init = c1

c2 = np.zeros((num_pc,num_cw,3))
for i in range ((num_pc)):
    for j in range(num_cw):
        r = np.random.random()
        if r < 0.33:
            c2[i][j][0] = 1
        elif r < 0.67:
            c2[i][j][1] = 1
        else :
            c2[i][j][2] = 1
c2_init = c2


hr = np.random.randint(25,100,size = num_cw)
for i in range(len(hr)):
    val = int(int(hr[i])*1000)
    hr[i] = val
hr_init = hr

num_vehicle1 = []
for i in range(len(data_new)):
    n = []
    n1 = np.random.randint(5,10)
    n2 = np.random.randint(15,25)
    n3 = np.random.randint(30,40)
        
    n.append(n1)
    n.append(n2)
    n.append(n3)
        
    num_vehicle1.append(n)  
    
num_vehicle2 = []
for i in range(len(data_new)):
    n = []
    n1 = np.random.randint(20,50)
    n2 = np.random.randint(30,60)
    n3 = np.random.randint(40,80)
    
    n.append(n1)
    n.append(n2)
    n.append(n3)
        
    num_vehicle2.append(n)


###################################
    #Run Here for Sensitivity analysis
###################################

epsilon = 2
print("NK Hybrid Clustering Start")
import time
start = time.time()
sol = NKCV2(data_new,size_pop,n_clusters,epsilon,k,num_iter,pc,n_ls)
end = time.time()
print("Algorithm time in Seconds:",end - start)
sol_copy_original = sol
loop = 0
count = 0
list1 = []
for i in sol :
    if i in list1:
        continue
    else:
        count+=1
        list1.append(i)
                    
print("Number of Clusters :",count)

cl = ['red','blue','green','yellow','black','purple','orange','violet','brown','pink','magenta','indigo']
for i in range(count):
    for j in range(len(data)):
        if sol[j] == list1[i]:
            if j in density_list:
                plt.scatter(data[j][0],data[j][1],marker='*',c = cl[i], s = 100)
            else:
                plt.scatter(data[j][0],data[j][1],c = cl[i],s = 40)
plt.show()            
                
                
counter = 0
Gep = Interaction_graph(k,data_new,epsilon)
while (counter < 100):
    loop = 0
    count = 0
    list1 = []
    for i in sol :
        if i in list1:
            continue
        else:
            count+=1
            list1.append(i)
                    
    
    density = Local_density(data_new,epsilon)
    prototype = cluster_proto(sol,count,density,list1)
    PC = np.zeros((count))
    for i in range(count):
        min_dist = 1000000
        index = 0
        for j in range(len(prototype)):
            if prototype[j][0] == list1[i]:
                proto_index = int(prototype[j][1])
        for j in range(len(data_new[0])):
            if data_new[proto_index][j] < min_dist:
                min_dist = data_new[proto_index][j]
                index = j
        PC[i] = density_list[index]
        
    
    dic_clust = {}
    for i in range(count):
        e = []
        for j in range(len(sol)):
            if sol[j] == list1[i]:
                e.append(j)
        dic_clust[i] = e
    
    
    density = Local_density(data_new,epsilon)
    prototype = cluster_proto(sol,count,density,list1)
    dic_pc = {}
    for i in range(count):
        e = []
        enot = []
        e1 = []
        min_dist = vil_pc_maxdistance
        for k in range(len(data_new[0])):
            e.append(density_list[k])
            for j in (dic_clust[i]):
                if data_new[j][k] > min_dist :
                    enot.append(density_list[k])
        for j in e:
            if j not in enot:
                e1.append(j)
                
        if len(e1) == 0:
            sol = cluster_split(data_new,data,sol,density,list1[i])
            loop = 1
            break
        dic_pc[i] = e1 
        
    if loop == 1 :
        counter += 1
    else:
        break
print("Number of Clusters :",count)
print()
#print(dic_pc)
print("Cluster Allocation")
print(dic_clust)

density = Local_density(data_new,epsilon)
prototype = cluster_proto(sol,count,density,list1)
dic_pc = {}
for i in range(count):
    e = []
    enot = []
    e1 = []
    min_dist = vil_pc_maxdistance
    for k in range(len(data_new[0])):
        e.append(density_list[k])
        for j in (dic_clust[i]):
            if data_new[j][k] > min_dist :
                enot.append(density_list[k])
    for j in e:
        if j not in enot:
            e1.append(j)
                
        dic_pc[i] = e1
print("Possible Procurement centres cluster wise:",dic_pc)
#print(sol_copy_original)
#sol = sol_copy_original
    
cl = ['red','blue','green','yellow','black','purple','orange','violet','brown','pink','magenta','indigo']
for i in range(count):
    for j in range(len(data)):
        if sol[j] == list1[i]:
            if i < 8:
                plt.scatter(data[j][0],data[j][1],marker='+',c = cl[i],s = 80)
            else:
                plt.scatter(data[j][0],data[j][1],marker='.',c = cl[i-8],s = 150)
plt.show()
#Procurement Center
J = 100000000    
pc = np.zeros((len(density_list),2))
for i in range(len(density_list)):
    index = density_list[i]
    pc[i][0] = df['Latitude'][index]
    pc[i][1] = df['Longitude'][index]

e2 = np.zeros((num_pc,num_cw))
for i in range(num_pc):
    c1 = ((pc[i][0],pc[i][1]))
    for j in range(num_cw):
        c2 = ((cw[j][0],cw[j][1]))
        d = (great_circle(c1,c2).km)
        e2[i][j] = d
        
#print(e2)

fc = fc_init

uc1 = [500, 800, 1000]
uc2 = [1500, 1800, 2000]
    
c1l = 20
c1m = 20*1.25
c1h = 20*1.5

c1 = c1_init
    
c2l = 20
c2m = 20*1.25
c2h = 20*1.5

c2 = c2_init 
      
e1 = data_new
df["Supply"] = df["Supply (Tonnes)"]
    
av = df["Supply"]
av = np.asarray(av)

hr = hr_init
        
omega1 = [5,8,10]
omega2 = [15,18,20]
    
A = [(i,j) for i in range(len(data)) for j in range(3) ]
B = [(i,j) for i in range(num_pc) for j in range(3) ]

    
s1 = [0.01, 0.016, 0.02]
s2 = [0.03, 0.036, 0.04]
    
count_sol = 1
for i in range(len(dic_pc)):
    count_sol = count_sol * len(dic_pc[i])

#print(dic_pc)
#print("Initialised")
print("Possible Solutions",count_sol)
solutions = []
count = 0
while count < count_sol:
    e = []
    for a in range(len(dic_pc)):
        r = np.random.randint(0,(len(dic_pc[a])))
        e.append(dic_pc[a][r])
    if e not in solutions:
        solutions.append(e)
        count += 1
        
        
#print(len(solutions))

#Integrated
min_cost = 10000000000000000000000000
for iterations in range(len(solutions)):
    if iterations%100 == 0:
        print("Cplex Iterations",iterations)
        
    X = np.zeros(num_pc) 
    for i in range(len(dic_pc)):
        pc = solutions[iterations][i]
        for j in range(len(density_list)):
            if pc == density_list[j] :
                X[j] = 1
        
    Y1 = np.zeros((len(data),num_pc))
    for i in range(len(dic_clust)):
        pc = solutions[iterations][i]
        for j in range(len(density_list)):
            if density_list[j] == pc :
                index = j
        for j in dic_clust[i]:
            Y1[j][index] = 1
                
    V = [(i,j) for i in range(len(data)) for j in range(num_pc) ]
    P = [(i,j) for i in range(num_pc) for j in range(num_cw)]
    
    W = np.zeros((len(data),num_pc))
    for i in range(len(Y1)):
        for j in range(len(Y1[i])):
            if Y1[i][j] == 1 :
                W[i][j] = df['Supply'][i]
                
    Y2 = np.zeros((num_pc,num_cw))
    for i in range(num_pc):
        min_dist = 10000
        for j in range(num_cw):
            if e2[i][j] <= min_dist:
                min_dist = e2[i][j]
                index_cw = j
        Y2[i][index_cw] = 1
        
        
    G = np.zeros((num_pc,num_cw))
    total = np.sum(W,axis = 0)
    for i in range(num_pc):
        for j in range(num_cw):
            if Y2[i][j] == 1:
                G[i][j] = total[i]

    
                         
    from docplex.mp.model import Model
    mdl = Model("model1")
    B1 = mdl.integer_var_dict(A,name = 'B1')
    B2 = mdl.integer_var_dict(B,name = 'B2')
        
    C = [(i) for i in range(len(density_list))]
    
    CO1 = [(i,j,k) for i in range(len(data)) for j in range(num_pc) for k in range(3)]
    CO2 = [(i,j,k) for i in range(num_pc) for j in range(num_cw) for k in range(3)]
    mdl.minimize(mdl.sum(sum(fc[i] * X[i] for i in C) + sum(uc1[j]*B1[i,j] for (i,j) in A) + sum((c1l*c1[i][j][0] + c1m*c1[i][j][1] + c1h*c1[i][j][2])*e1[i][j]*W[i][j] for (i,j) in V) + sum(uc2[j]*B2[i,j] for (i,j) in B) + sum((c2l*c2[i][j][0] + c2m*c2[i][j][1] + c2h*c2[i][j][2])*e2[i][j]*G[i][j] for (i,j) in P) + sum(e1[i][j]*lamda*s1[k]*B1[i,k]for (i,j,k) in CO1 )+ sum(e2[i][j]*lamda*s2[k]*B2[i,k]for (i,j,k) in CO2 ) + sum(W[i][j]*pi for (i,j) in V) + sum(G[i][j]*pi for (i,j) in P) + sum(e2[i][j]*lamda*s2[k]*B2[i,k]for (i,k) in B )))
        
    mdl.add(B1[i,j] <= num_vehicle1[i][j] for (i,j) in A)
    mdl.add(B2[i,j] <= num_vehicle2[i][j] for (i,j) in B)
    
    mdl.add((sum(omega1[j] * B1[i,j] for j in range(3)) >= df['Supply'][i] for i in range(len(data))))
    mdl.add((sum(omega2[j] * B2[i,j] for j in range(3)) >= total[i] for i in range(num_pc)))
    
    sol1 = mdl.solve()
    cost = sol1.get_objective_value()
    if cost <= min_cost:
        min_cost = cost
        min_sol = solutions[iterations]
        index = iterations
#print(min_sol)

print("Final Cluster Allocation")
for i in range(len(min_sol)):
    for j in range(len(density_list)):
        if min_sol[i] == density_list[j]:
            print("Cluster :",i,"Procurement Centr :",j)

X = np.zeros(num_pc) 
iterations = index
for counter1 in range(1):
    for i in range(len(dic_pc)):
        pc = min_sol[i]
        for j in range(len(density_list)):
            if pc == density_list[j] :
                X[j] = 1
        
    Y1 = np.zeros((len(data),num_pc))
    for i in range(len(dic_clust)):
        pc = min_sol[i]
        for j in range(len(density_list)):
            if density_list[j] == pc :
                index = j
        for j in dic_clust[i]:
            Y1[j][index] = 1
                
    V = [(i,j) for i in range(len(data)) for j in range(num_pc) ]
    P = [(i,j) for i in range(num_pc) for j in range(num_cw)]
    
    W = np.zeros((len(data),num_pc))
    for i in range(len(Y1)):
        for j in range(len(Y1[i])):
            if Y1[i][j] == 1 :
                W[i][j] = df['Supply'][i]
                
    Y2 = np.zeros((num_pc,num_cw))
    for i in range(num_pc):
        min_dist = 10000
        for j in range(num_cw):
            if e2[i][j] <= min_dist:
                min_dist = e2[i][j]
                index_cw = j
        Y2[i][index_cw] = 1
        
        
    G = np.zeros((num_pc,num_cw))
    total = np.sum(W,axis = 0)
    #print(total)
    for i in range(num_pc):
        for j in range(num_cw):
            if Y2[i][j] == 1:
                G[i][j] = total[i]

# Integrated
    mdl2 = Model("model")
    B1 = mdl2.integer_var_dict(A,name = 'B1')
    B2 = mdl2.integer_var_dict(B,name = 'B2')
        
    C = [(i) for i in range(len(density_list))]
    
    CO1 = [(i,j,k) for i in range(len(data)) for j in range(num_pc) for k in range(3)]
    CO2 = [(i,j,k) for i in range(num_pc) for j in range(num_cw) for k in range(3)]
    mdl2.minimize(mdl2.sum(sum(fc[i] * X[i] for i in C) + sum(uc1[j]*B1[i,j] for (i,j) in A) + sum((c1l*c1[i][j][0] + c1m*c1[i][j][1] + c1h*c1[i][j][2])*e1[i][j]*W[i][j] for (i,j) in V) + sum(uc2[j]*B2[i,j] for (i,j) in B) + sum((c2l*c2[i][j][0] + c2m*c2[i][j][1] + c2h*c2[i][j][2])*e2[i][j]*G[i][j] for (i,j) in P) + sum(e1[i][j]*lamda*s1[k]*B1[i,k]for (i,j,k) in CO1 )+ sum(e2[i][j]*lamda*s2[k]*B2[i,k]for (i,j,k) in CO2 ) + sum(W[i][j]*pi for (i,j) in V) + sum(G[i][j]*pi for (i,j) in P) + sum(e2[i][j]*lamda*s2[k]*B2[i,k]for (i,k) in B )))
        
    mdl2.add(B1[i,j] <= num_vehicle1[i][j] for (i,j) in A)
    mdl2.add(B2[i,j] <= num_vehicle2[i][j] for (i,j) in B)
    
    mdl2.add((sum(omega1[j] * B1[i,j] for j in range(3)) >= df['Supply'][i] for i in range(len(data))))
    mdl2.add((sum(omega2[j] * B2[i,j] for j in range(3)) >= total[i] for i in range(num_pc)))
    
    
    sol1 = mdl2.solve()
    val = mdl2.solution.get_value_dict(B1)
    B1 = val
    val = mdl2.solution.get_value_dict(B2)
    B2 = val
    cost = sol1.get_objective_value()
    e_cost = sum(fc[i] * X[i] for i in C)
    i_cost = sum(W[i][j]*pi for (i,j) in V) + sum(G[i][j]*pi for (i,j) in P)
    t_cost = sum(uc1[j]*B1[i,j] for (i,j) in A) + sum((c1l*c1[i][j][0] + c1m*c1[i][j][1] + c1h*c1[i][j][2])*e1[i][j]*W[i][j] for (i,j) in V) + sum(uc2[j]*B2[i,j] for (i,j) in B) + sum((c2l*c2[i][j][0] + c2m*c2[i][j][1] + c2h*c2[i][j][2])*e2[i][j]*G[i][j] for (i,j) in P)
    env_cost = sum(e1[i][j]*lamda*s1[k]*B1[i,k]for (i,j,k) in CO1 )+ sum(e2[i][j]*lamda*s2[k]*B2[i,k]for (i,j,k) in CO2 )
    

    print()
    print("Supply matrix from V to PC")
    print(W)
    print()
    print("Supply matrix from PC to CW")
    print(G)
    print()
    print("Capacity Matrix of each PC")
    print(np.sum(W,axis = 0))
    print()
    print("Trucks used from PC to CW where (i,j) denote ith PC and jth Vehicle type")
    print(B2)
    print()
    print('Establishment Cost:',e_cost)
    print('Inventory cost:',i_cost)
    print('Transportation Cost:',t_cost)
    print('Environment_cost',env_cost)
    print('Total cost :',cost)