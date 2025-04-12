import os
import time
import random
import numpy as np

import loaddata
import evaluation
from MyOrder import KeepOrder

class KSUMS:
    def __init__(self, NN, NND, c_true, debug, max_dd):
        self.N = len(NN)
        self.c_true = c_true
        self.debug = debug
        self.NN = NN
        self.NND = NND

        self.y = [-1] * self.N
        self.time_arr = [] #= np.empty(self.N, dtype=np.float64)  # record time cost in one opt
        self.sigma = 0      # may not use

        self.t = np.zeros(self.c_true, dtype=np.float64)
        self.v = np.zeros(self.c_true, dtype=bool)  # visited
        self.u = np.zeros(self.c_true, dtype=int)   # the number of points in cluster
        # self.knn2c = [0] * self.N   # may not needed as global
        self.Y = [] #= np.empty((0, self.N), dtype=np.int64)
        
        if max_dd >= 0:
            self.max_d = max_dd
        else:
            self.max_d = np.amax(NND)
        
        if self.debug:
            print(f"max_distance is {self.max_d}")
            print(f"sample number = {self.N}, with {self.c_true} clusters")

        self.symmetry(NN, NND)

        self.KO = None

    def init(self):
        self.y = [-1] * self.N
        n = [0] * self.N
        n_up = self.N // self.c_true    # upper bound of cluster points

        num_clu = 0     # number of cluster
        
        for i in range(self.N):
            if self.y[i] == -1:     # not labeled yet
                tmpL = [i]
                self.y[i] = num_clu
                n[num_clu] += 1
                flag = False        # denote the cluster is full

                while tmpL and not flag:
                    current_id = tmpL.pop(0)
                # I think here should iterate current_id's neighbor
                    for neighbor in self.NN[current_id]:   # iterate every neighbor
                        if self.y[neighbor] == -1:  # no cluster yet
                            tmpL.append(neighbor)
                # something wrong here, if current_id is not used, why 
                # this for-loop exist in while-loop? it'll be run only 
                # once. and the tmpL also make no sense.
                            self.y[neighbor] = num_clu
                            n[num_clu] += 1
                            if n[num_clu] >= n_up:
                                flag = True     # time to end
                                tmpL.clear()
                                break
                num_clu += 1

        if self.debug:
            print(f"num_clu = {num_clu} (c = {self.c_true})")

        if num_clu > self.c_true:
            self.KO = KeepOrder(self.y, self.N, num_clu)
            old2new = [-1] * num_clu

            # merge small clusters
            for i in range(num_clu - self.c_true):
                old_clu = self.KO.o2c[i]
                old2new[old_clu] = random.randint(0, self.c_true-1)

            # preserve large cluster and re-number
            new_idx = 0
            for i in range(num_clu - self.c_true, num_clu):
                old_clu = self.KO.o2c[i]
                old2new[old_clu] = new_idx
                new_idx += 1

            # update label(cluster number)
            for i in range(self.N):
                if self.y[i] != -1:
                    self.y[i] = old2new[self.y[i]]

    def symmetry(self, NN, NND):
        if self.debug:
            print(f"NN is ndarray? {isinstance(NN, np.ndarray)}")
            print(f"NND is ndarray? {isinstance(NND, np.ndarray)}")
        
        if isinstance(NN, np.ndarray):
            NN = NN.tolist()
        if isinstance(NND, np.ndarray):
            NND = NND.tolist()
        
        N = len(NN)
        if N == 0:
            return NN, NND

        # reverse_map: { node j : [(i, distance)] }
        reverse_map = [[] for _ in range(N)]
        for i in range(N):
            for j, d in zip(NN[i], NND[i]):
                reverse_map[j].append((i, d))

        # symmetry
        cnt = 0
        for i in range(N):
            original_neighbors = set(NN[i])  # original_neighbors set
            # add the reverse neighbor that not in original_neighbors
            for j, d in reverse_map[i]:
                if j not in original_neighbors:
                    NN[i].append(j)
                    NND[i].append(d)
                    cnt += 1
        if self.debug:
            print(f"symmetry add {cnt} times")
        
        NN_np = np.array([np.array(node, dtype=np.int32) for node in NN], dtype=object)
        NND_np = np.array([np.array(node, dtype=np.float64) for node in NND], dtype=object)
        return NN_np, NND_np
    
    def opt_once(self, MAX_ITER, our_init):
        if our_init == 1:
            self.init()
        else:
            # random cluster
            self.y = np.random.randint(0, self.c_true, self.N).tolist()

        self.KO = KeepOrder(self.y, self.N, self.c_true)

        start_time = time.time()
        converge = True

        # iterate MAX_ITER times
        for _ in range(MAX_ITER):
            converge = True
            for sam_i in range(self.N):
                c_old = self.y[sam_i]
                c_new = self.find_c_new(sam_i)

                if c_new != c_old:
                    converge = False
                    self.y[sam_i] = c_new
                    self.KO.sub(self.KO.c2o[c_old])
                    self.KO.add(self.KO.c2o[c_new])

            if converge:
                break

        return time.time() - start_time
    
    # do the clustering {rep} times, each time has {MAX_ITER} iterations
    def opt(self, rep, MAX_ITER, our_init): 
        # self.time_arr.reshape(rep)
        # self.Y.reshape(rep)
        # self.time_arr = np.empty(rep)
        # self.Y = np.empty((rep, self.N))
        for i in range(rep):
            time_once = self.opt_once(MAX_ITER, our_init)
            self.time_arr.append(time_once)
            current_y = np.array(self.y, dtype=np.int32) 
            self.Y.append(current_y)
        self.Y = np.stack(self.Y)
        self.time_arr = np.stack(self.time_arr)

    def find_c_new(self, sam_i):    # find a new cluster for sam_i
        # get the label of all neighbor
        # knn2c is the cluster of the neighbor
        knn2c = np.array([self.y[nb] for nb in self.NN[sam_i]], dtype=int)
        # unique_clusters is all the cluster that shows in neighbor 
        unique_clusters = np.unique(knn2c)
        
        # initiate
        self.t = np.zeros(self.c_true, dtype=np.float64)
        self.u = np.zeros(self.c_true, dtype=int)
        self.v = np.zeros(self.c_true, dtype=bool)
        
        # calculate distance and count(vectorize)
        distances = np.array(self.NND[sam_i])
        # np,add.at(X, a, num) means add num to X[a].
        np.add.at(self.t, knn2c, distances)
        np.add.at(self.u, knn2c, 1)
        
        # for the un-visited part, the cost will be k*γ
        if len(unique_clusters) > 0:
            # all the clusters that appeared in i's neighbor
            for cluster in unique_clusters:
                if not self.v[cluster]: # if not visited yet
                    self.v[cluster] = True
                    # get the order of cluster
                    order_idx = self.KO.c2o[cluster]
                    tmp_ni = self.KO.o2ni[order_idx]
                    # calculate t value
                    self.t[cluster] += (tmp_ni - self.u[cluster]) * self.max_d

        ## find the minimum 
        # valid_clusters = knn2c[np.isin(knn2c, unique_clusters)]
        # if len(valid_clusters) == 0:    # actually impossiable
        #     return self.KO.o2c[0]
        
        ## get the cluster number with minimum t-value, get min t(1)
        candidate_clusters = np.unique(knn2c)   # list of all the cluster 
        min_val = self.t[candidate_clusters].min()
        min_cluster = candidate_clusters[np.argmin(self.t[candidate_clusters])]

        # compare t(1) and t(2), t(2) is the first of KeepOrder
        if self.KO.o2ni[0] * self.max_d < min_val:
            return self.KO.o2c[0]
        else:
            return min_cluster


if __name__ == '__main__':
    data_name = "FaceV5"
    X, y_true, c_true, NN, NND, t1 = loaddata.load_data(data_name)

    obj = KSUMS(NN.astype(np.int32), NND, c_true, debug=True, max_dd=-1)
    obj.opt(rep=10, MAX_ITER=100, our_init=1)
    t2 = obj.time_arr

    acc = evaluation.multi_accuracy(y_true, obj.Y)
    nmi = evaluation.multi_nmi(y_true, obj.Y)
    ari = evaluation.multi_ari(y_true, obj.Y)
    print(f"{data_name}: {np.mean(acc):.3f}(±{np.std(acc):.2e}), {np.mean(nmi):.3f}(±{np.std(nmi):.2e}), {np.mean(ari):.3f}(±{np.std(ari):.2e}), {t1 + np.mean(t2):.3f}(±{t1 + np.std(t2):.2e})")
    #  0.949(2.15e-03), 0.979(1.67e-03), 0.844(3.15e-02), 1.955(1.49e+00)
    # paper: 0.963, 0.986, 0.915, 0.254