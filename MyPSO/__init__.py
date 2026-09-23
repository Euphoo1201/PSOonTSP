"""
MyPSO —— 一个基于连续位置向量的粒子群优化 (Particle Swarm Optimization) 实现。

公开接口:
    MyPSO    : 粒子群优化器, 负责初始化种群、迭代更新与全局最优维护
    Particle : 单个粒子, 负责自身速度/位置更新与个体最优维护

用法:
    from MyPSO import MyPSO

    def f(x):
        return -float(((x - 0.5) ** 2).sum())   # 注意: 目标函数越大越好

    pso = MyPSO(n_max_iterations=100, n_particles=30, n_dimensions=2, f_target=f)
    pso.pso_process()
    print(pso.g_best, f(pso.g_best))
"""

from .core import MyPSO, Particle

__all__ = ["MyPSO", "Particle"]
