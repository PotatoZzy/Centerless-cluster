class KeepOrder:
    class LRInd:
        def __init__(self):
            self.l = -1
            self.r = -1

    def __init__(self, y, N, c):
        """
        :param y: list of clusters，each element represents the number of sample's cluster
        :param N: total number of samples
        :param c: total number of clusters
        """
        self.N = N
        self.c = c
        
        # some mapping list 
        self.o2ni = [0] * c  # order to number of cluster
        self.o2c = list(range(c))  # order to cluster by increasing sequence
        self.c2o = [0] * c  # cluster to order
        self.ni2o = [self.LRInd() for _ in range(N)]  # cluster size to order

        # counting cluster size
        for cluster in y:
            self.o2ni[cluster] += 1

        # sort by cluster size
        # get cluster-by-order(o2c)
        self.o2c = sorted(range(c), key=lambda x: self.o2ni[x])
        # update o2ni as order size
        self.o2ni.sort()

        # get the mapping from cluster number to order
        for order, cluster in enumerate(self.o2c):
            self.c2o[cluster] = order

        # mapping from cluster size to other
        current_size = -1
        for i in range(c):
            if self.o2ni[i] != current_size:
                if current_size != -1:
                    self.ni2o[current_size].r = i-1
                current_size = self.o2ni[i]
                self.ni2o[current_size].l = i
            if i == c-1:
                self.ni2o[current_size].r = i

    def sub(self, order_id):
        """ reduce 1 in the place of order_id """
        old_size = self.o2ni[order_id]
        
        # get the original minimum
        old_l = self.ni2o[old_size].l
        
        # update cluster size
        self.ni2o[old_size].l += 1
        if self.ni2o[old_size].l > self.ni2o[old_size].r:
            self.ni2o[old_size].l = self.ni2o[old_size].r = -1
            
        # update right bound of new size
        new_size = old_size - 1
        self.ni2o[new_size].r = old_l
        if self.ni2o[new_size].l == -1:
            self.ni2o[new_size].l = old_l
            
        # exchange the cluster sequence
        self.o2ni[order_id] -= 1
        self.o2c[order_id], self.o2c[old_l] = self.o2c[old_l], self.o2c[order_id]
        
        # update reverse mapping(cluster to order)
        self.c2o[self.o2c[order_id]] = order_id
        self.c2o[self.o2c[old_l]] = old_l

    def add(self, order_id):
        """ add 1 in the place of order_id """
        old_size = self.o2ni[order_id]
        
        # Almost same process of sub
        old_r = self.ni2o[old_size].r
        
        self.ni2o[old_size].r -= 1
        if self.ni2o[old_size].r < self.ni2o[old_size].l:
            self.ni2o[old_size].l = self.ni2o[old_size].r = -1
            
        # update left bound of new size
        new_size = old_size + 1
        self.ni2o[new_size].l = old_r
        if self.ni2o[new_size].r == -1:
            self.ni2o[new_size].r = old_r
            
        self.o2ni[order_id] += 1
        self.o2c[order_id], self.o2c[old_r] = self.o2c[old_r], self.o2c[order_id]
        
        self.c2o[self.o2c[order_id]] = order_id
        self.c2o[self.o2c[old_r]] = old_r

if __name__ == "__main__":
    # Test example
    y = [0, 1, 2, 0, 1]
    ko = KeepOrder(y, N=5, c=5)
    
    print("original state:")
    print("o2ni (order to cluster size):", ko.o2ni)  # [1, 2, 2]
    print("o2c (order to cluster number):", ko.o2c)   # [2, 0, 1]
    print("c2o (cluster num to order):", ko.c2o)   # [1, 2, 0]
    
    # test sub func
    ko.sub(1)
    print("\nAfter reducing cluster-0:")
    print("o2ni:", ko.o2ni)        # [1, 1, 2]
    print("o2c:", ko.o2c)         # [2, 1, 0]