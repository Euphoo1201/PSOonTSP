import numpy as np

import MyPSO


#管理TSP问题构建所需的所有数据结构
class City:
    def __init__(self,idx:int,adj_dists:dict[int, float] | None = None):
        self.idx = idx
        if adj_dists is not None:
            self.adj_dists = adj_dists 
        else:
            self.adj_dists = {}    #邻接城市以及边的权重

    def connect_to(self,idx:int,weight:float):
        self.adj_dists[idx] = weight

class TSP:
    def __init__(self,start_idx:int, n_cities:int):
        self.start_idx = start_idx      #触发城市的索引
        self.n_cities = n_cities        #城市个数
        self.cities:dict[int,City] = {}
        self.b_has_built = False


    def build_map(self,cities:np.ndarray,is_digraph:bool):
        """
        构建地图
        :param cities: n*3矩阵，每一行[city1,city2,weight]
        :return:
        """
        for i in range(self.n_cities):
            self.cities[i] = City(i)

        for i in range(cities.shape[0]):
            city1 = int(cities[i,0])
            city2 = int(cities[i,1])
            weight = float(cities[i,2])
            if city1 not in self.cities or city2 not in self.cities:
                raise ValueError(f"地图数据与预设城市编号不符")
            self.cities[city1].connect_to(city2,weight)
            if not is_digraph:
                self.cities[city2].connect_to(city1,weight)
        if self.start_idx not in self.cities:
            raise ValueError(f"起点 {self.start_idx} 不在图中")
        self.b_has_built = True

        # TSP.build_map 末尾
        w = np.array([weight for c in self.cities.values() for weight in c.adj_dists.values()])
        self.max_tour_bound = float(np.sort(w)[-self.n_cities:].sum())   # S：前 n 大边权之和


#将TSP问题建模成可以供PSO处理的形式
class TSPManager:
    def __init__(self,tsp_inst:TSP,punish_rate:float = 2.0):
        if tsp_inst.b_has_built is False:
            raise ValueError("TSP实例尚未构建地图，请先调用build_map方法")
        self.tsp_inst = tsp_inst    
        self.penalty = tsp_inst.max_tour_bound * punish_rate                 #增大对于路径无效的惩罚力度提供的惩罚系数

    def from_float_score_to_int_path(self,scores:np.ndarray) -> list:
        """
        将传入的浮点数顺序打分，从小到大映射成为n个城市的访问次序
        :return:表示访问次序，例如输入的打分是[0.1,0.5,0.15]，则得到三座城市的顺序：[0，2，1]，表示先访问第一城，其次访问第三城，最后访问第二城
        """
        l_scores:list = []
        for i in scores:
            l_scores.append(i)
        l_path:list = []
        n_size= len(l_scores)
        for i in range(n_size):
            cur = l_scores.index(min(l_scores))
            l_path.append(cur)
            l_scores[cur] += max(l_scores)-min(l_scores) + 1   #将已经访问过的城市的分数增大到大于最大值，使得给定for循环次数内不会再访问
        l_path.append(self.tsp_inst.start_idx)  #由于编码对策特性，需要在结尾补充回到起点的这一步
        return l_path

    def check_path_connectivity(self,path:list) -> int:
        """
        检查给定访问顺序是否都有边连通
        :return:不连通的次数
        """
        n_disconn_cnt = 0         
        #检查起点是否正确
        if path[0] != self.tsp_inst.start_idx:
            n_disconn_cnt += 1
        #检查路径中是否有不连通的边
        for i in range(len(path) - 1):
            if path[i] in self.tsp_inst.cities:
                if path[i+1] not in self.tsp_inst.cities[path[i]].adj_dists:
                    n_disconn_cnt += 1
        return n_disconn_cnt

    def calculate_path_weight(self,path:list):
        """
        计算路程总长度
        :return:总长度
        """
        total_dist:float = 0.0
        for i in range(len(path) - 1):
            cur = path[i]
            nxt = path[(i+1)]
            if cur in self.tsp_inst.cities:
                if nxt in self.tsp_inst.cities[cur].adj_dists:
                    total_dist += self.tsp_inst.cities[cur].adj_dists[nxt]
        return total_dist

    def get_target_func(self):
        """
        获取目标函数
        :return:
        """
        def f(x:np.ndarray):
            """
            目标函数，输入为一个连续位置向量，输出为该位置向量对应的TSP路径的总权重
            :param x: 输入的是pos，这些pos维度每个维度对应一座城市，不同维度大小决定先后
            :return:如果此向量给定顺序可以实现，则给一个随着总路径变长而减小的正值；如果无法构成路线，则返回惩罚性负值
            """
            if len(x) != self.tsp_inst.n_cities:
                raise ValueError(f"输入向量维度 {len(x)} 与城市数量 {self.tsp_inst.n_cities} 不符，请检查输入")
            path = self.from_float_score_to_int_path(x)
            total_dist = self.calculate_path_weight(path)
            cnt_disconn = self.check_path_connectivity(path)
            return -(total_dist + self.penalty * cnt_disconn)    
        return f

