class KeepOrder:
    class LRInd:
        def __init__(self):
            self.l = -1
            self.r = -1

    def __init__(self, y, N, c):
        """
        :param y: 簇标签列表，每个元素表示对应样本所属簇的编号
        :param N: 样本总数
        :param c: 簇总数
        """
        self.N = N
        self.c = c
        
        # 初始化数据结构
        self.o2ni = [0] * c  # 按顺序排列的簇大小列表
        self.o2c = list(range(c))  # 按簇大小升序排列的簇索引
        self.c2o = [0] * c  # 簇编号到排序位置的映射
        self.ni2o = [self.LRInd() for _ in range(N)]  # 簇大小对应的索引范围

        # 步骤1: 统计初始簇大小
        for cluster in y:
            self.o2ni[cluster] += 1

        # 步骤2: 对簇按大小排序
        # 生成排序后的簇索引 (o2c)
        self.o2c = sorted(range(c), key=lambda x: self.o2ni[x])
        # 同步更新o2ni为排序后的簇大小
        self.o2ni.sort()

        # 步骤3: 建立簇到排序位置的映射
        for order, cluster in enumerate(self.o2c):
            self.c2o[cluster] = order

        # 步骤4: 建立簇大小到索引范围的映射
        current_size = -1
        start = 0
        for i in range(c):
            if self.o2ni[i] != current_size:
                if current_size != -1:
                    self.ni2o[current_size].r = i-1
                current_size = self.o2ni[i]
                self.ni2o[current_size].l = i
            if i == c-1:
                self.ni2o[current_size].r = i

    def sub(self, order_id):
        """ 减少指定顺序位置簇的大小 """
        old_size = self.o2ni[order_id]
        
        # 获取该大小的最左索引
        old_l = self.ni2o[old_size].l
        
        # 更新大小映射
        self.ni2o[old_size].l += 1
        if self.ni2o[old_size].l > self.ni2o[old_size].r:
            self.ni2o[old_size].l = self.ni2o[old_size].r = -1
            
        # 更新新大小的右边界
        new_size = old_size - 1
        self.ni2o[new_size].r = old_l
        if self.ni2o[new_size].l == -1:
            self.ni2o[new_size].l = old_l
            
        # 交换簇顺序
        self.o2ni[order_id] -= 1
        self.o2c[order_id], self.o2c[old_l] = self.o2c[old_l], self.o2c[order_id]
        
        # 更新反向映射
        self.c2o[self.o2c[order_id]] = order_id
        self.c2o[self.o2c[old_l]] = old_l

    def add(self, order_id):
        """ 增加指定顺序位置簇的大小 """
        old_size = self.o2ni[order_id]
        
        # 获取该大小的最右索引
        old_r = self.ni2o[old_size].r
        
        # 更新大小映射
        self.ni2o[old_size].r -= 1
        if self.ni2o[old_size].r < self.ni2o[old_size].l:
            self.ni2o[old_size].l = self.ni2o[old_size].r = -1
            
        # 更新新大小的左边界
        new_size = old_size + 1
        self.ni2o[new_size].l = old_r
        if self.ni2o[new_size].r == -1:
            self.ni2o[new_size].r = old_r
            
        # 交换簇顺序
        self.o2ni[order_id] += 1
        self.o2c[order_id], self.o2c[old_r] = self.o2c[old_r], self.o2c[order_id]
        
        # 更新反向映射
        self.c2o[self.o2c[order_id]] = order_id
        self.c2o[self.o2c[old_r]] = old_r

# 示例用法
if __name__ == "__main__":
    # 示例数据：3个簇，5个样本
    y = [0, 1, 2, 3, 4]  # 簇0有2个样本，簇1有2个样本，簇2有1个样本
    ko = KeepOrder(y, N=5, c=5)
    
    print("初始状态:")
    print("o2ni (簇大小):", ko.o2ni)  # [1, 2, 2]
    print("o2c (簇索引):", ko.o2c)   # [2, 0, 1]
    print("c2o (簇位置):", ko.c2o)   # [1, 2, 0]
    
    # 测试减少操作
    ko.sub(1)  # 减少第二个位置的簇（原簇0）
    print("\n减少簇0大小后:")
    print("o2ni:", ko.o2ni)        # [1, 1, 2]
    print("o2c:", ko.o2c)         # [2, 1, 0]