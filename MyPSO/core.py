import random
import numpy as np
from typing import Callable
from . import utils


class Particle:
    def __init__(self,pos:np.ndarray,
                 vel:np.ndarray,
                 f_target:Callable[[np.ndarray], float],
                 max_vel:float):
        self.pos = pos
        self.vel = vel
        self.p_best = None
        self.f_target = f_target
        self.max_vel = max_vel

    def target_func(self, pos:np.ndarray)->float:
        return self.f_target(pos)

    def update(self,
               g_best:np.ndarray,
               omega:float,
               c1: float,
               c2: float,
               r1: float,
               r2: float,
               ):
        """
        单点移动
        """
        if self.p_best is None:
            self.p_best = np.copy(self.pos)

        #速度更新
        self.vel = self.vel * omega + c1*r1*(self.p_best - self.pos)+c2*r2*(g_best - self.pos)
        if np.linalg.norm(self.vel) > self.max_vel:#超过最大限制速度则限速
            self.vel = self.vel / np.linalg.norm(self.vel) * self.max_vel   
        
        #位置更新
        self.pos = self.pos + self.vel
        
        if self.target_func(self.pos) > self.target_func(self.p_best):
            self.p_best = np.copy(self.pos)


class MyPSO:
    def __init__(self,
                 n_max_iterations:int,
                 n_particles:int,
                 n_dimensions:int,
                 f_target: Callable[[np.ndarray], float],
                 c1:float = 2,
                 c2:float = 2,
                 omega_min:float = 0.4,
                 omega_max:float = 0.7,
                 max_vel:float = 0.5):
        self.n_max_iterations = n_max_iterations
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.population = []
        self.omega_max = omega_max      #最大惯性权重
        self.omega_min = omega_min      #最小惯性权重
        self.omega = omega_max  # 惯性权重,在max与min之间做线性插值
        self.c1 = c1  # 自我认知学习因子
        self.c2 = c2  # 群体学习因子
        self.r1 = None  # 自我随机数
        self.r2 = None  # 群体随机数
        self.f_target = f_target    #目标函数，由业务逻辑提供
        self.max_vel = max_vel      #最大速度限制
        self.g_best = None      #所有粒子搜索历史的最优情况
        self.n_iter = 1         #迭代次数，从第一次开始


    def init_population(self):
        """
        初始化粒子群，单个粒子为n_dimensions维的向量
        :return:
        """
        #清空上次执行脏数据
        self.g_best = None
        self.n_iter = 1
        if self.population is not None:
            self.population.clear()
        for i in range(self.n_particles):
            particle = Particle(np.random.random(self.n_dimensions),
                                np.random.random(self.n_dimensions),
                                self.f_target,
                                self.max_vel)
            #初始化全局最佳pos
            if self.g_best is None:
                self.g_best = particle.pos.copy()
            elif self.f_target(self.g_best) < self.f_target(particle.pos):
                self.g_best = particle.pos.copy()
            self.population.append(particle)

    def update_omega(self):
        """
        惯性权重更新，由大变小
        """
        self.omega = self.omega_max - (self.omega_max - self.omega_min) * (self.n_iter / self.n_max_iterations)

    def update_population(self):
        """
        粒子群的一次更新
        :return:
        """
        if not self.population:
            print("粒子群为空")
            return
        if self.g_best is None:
            print("最优粒子位置不能为空")
            return
        if self.f_target is None:
            print("未提供有效的目标函数")
            return
        self.r1 = np.random.rand(self.n_particles)
        self.r2 = np.random.rand(self.n_particles)#  每个粒子的随机数不能相同，保证各自的变化姿态独立
        for i in range(self.n_particles):
            self.population[i].update(self.g_best,
                                      self.omega,
                                      c1=self.c1,
                                      c2=self.c2,
                                      r1=self.r1[i],
                                      r2=self.r2[i])
        self.update_omega()
        #更新g_best
        for particle in self.population:
            if self.f_target(particle.p_best) > self.f_target(self.g_best):
                self.g_best = particle.p_best.copy()
        #更新迭代次数
        self.n_iter += 1


    def pso_process(self):
        """
        完整的一次pso流程
        :return:
        """
        self.init_population()
        for i in range(self.n_max_iterations):
            self.update_population()

