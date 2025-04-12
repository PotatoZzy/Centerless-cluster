import numpy as np
import time

class KSUMX:
    def __init__(self, X: np.ndarray, c_true: int, debug: int = 0):
        self.X = X.T  # Convert to (dim, N) format
        self.dim, self.N = self.X.shape
        self.c_true = c_true
        self.debug = debug
        # self.xnorm = None
        self.xnorm = np.square(self.X).sum(axis=0)
        self.S = np.zeros((c_true, self.dim))
        self.n = np.zeros(c_true)
        self.v = np.zeros(c_true)

        self.Y = None
        self.time_arr = []
        self.iter_arr = []
        if self.debug:
            print(f"c_true = {c_true}")
            print(f"X shape: {self.X.shape}")
            print(f"xnorm shape: {self.xnorm.shape}")
        

    def _init_y(self, idx: int):
        # y = self.Y[idx]
        self.Y[idx] = np.random.randint(0, self.c_true, self.N)
        y = self.Y[idx]
        one_hot = np.zeros((self.N, self.c_true), dtype=bool)
        one_hot[np.arange(self.N), y] = 1
        self.n = one_hot.sum(axis=0)        # (c_true)
        self.S = (one_hot.T @ self.X.T)   # (c_true, dim)
        self.v = one_hot.T @ self.xnorm     # (c_true,)
        # if self.debug:
        #     print(f"n shape: {self.n.shape}")
        #     print(f"S shape: {self.S.shape}")
        #     print(f"v shape: {self.v.shape}")

    
    def _update_clusters(self, Yidx: int, max_iter: int) -> int:
        y = self.Y[Yidx]
        for iter_num in range(max_iter):
            total_changes = 0
            order = np.random.permutation(self.N)  # random sequence to avoid problems
            
            for idx in order:
                x_i = self.X[:, idx]          # (dim,)
                xnorm_i = self.xnorm[idx]
                
                # calculate distance from this point to all clusters
                #  t_k = n * X_i + sigma(||x_j||^2) - 2 * sigma(x_j) [in S] 
                t = self.n * xnorm_i + self.v - 2 * self.S.dot(x_i)
                # the new cluster is the number with minimum t value
                new_c = np.argmin(t)
                old_c = y[idx]

                if old_c != new_c:
                    # update n, v, S, y
                    self.n[old_c] -= 1
                    self.n[new_c] += 1
                    self.v[old_c] -= xnorm_i
                    self.v[new_c] += xnorm_i
                    self.S[old_c] -= x_i
                    self.S[new_c] += x_i
                    
                    y[idx] = new_c
                    total_changes += 1

            # check if converge
            if total_changes < self.N // 1000:
                return iter_num + 1
        self.Y[Yidx] = y
        return max_iter

    def opt(self, Y: np.ndarray, max_iter: int):
        self.Y = Y
        rep = len(Y)
        if self.debug:
            print(f"rep = {rep}")
        
        for rep_idx in range(rep):
            y = Y[rep_idx]
            start_time = time.time()
            
            # Initialization
            self._init_y(rep_idx)     # not change y
            
            # Clustering iterations
            # iter_count = self._update_clusters(y, block_size, max_iter)
            iter_count = self._update_clusters(rep_idx, max_iter)
            
            # Record performance
            self.iter_arr.append(iter_count)
            self.time_arr.append(time.time() - start_time)
            if self.debug:
                print(f"Running {rep_idx}: cost {self.time_arr[rep_idx]}s")

import evaluation
import loaddata

data_name = "FaceV5"
X, y_true, c_true, NN, NND, t1 = loaddata.load_data(data_name)
obj = KSUMX(X, c_true, debug=True)

init_Y = np.zeros((10, X.shape[0]), dtype=int)
obj.opt(init_Y, max_iter=100)

acc = evaluation.multi_accuracy(y_true, obj.Y)
nmi = evaluation.multi_nmi(y_true, obj.Y)
ari = evaluation.multi_ari(y_true, obj.Y)
print(obj.Y[0])
tt = obj.time_arr
print(f"{data_name}: {np.mean(acc):.3f}(±{np.std(acc):.2e}), {np.mean(nmi):.3f}(±{np.std(nmi):.2e}), {np.mean(ari):.3f}(±{np.std(ari):.2e}), {np.mean(tt):.3f}(±{t1 + np.std(tt):.2e})")
# paper: 0.963, 0.986, 0.915, 0.254
# our: 0.965(±3.08e-03), 0.986(±8.69e-04), 0.919(±4.10e-03), 0.750(±1.42e+00)