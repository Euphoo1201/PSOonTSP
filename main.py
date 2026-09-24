from MyPSO import MyPSO
from TSP import TSP,TSPManager
import numpy as np

#问题参数配置
START_IDX = 0
N_CITIES = 5
#TSP建模参数配置
PUNISH_RATE = 2
#PSO参数配置
N_MAX_ITERATION = 50
N_PARTICLES     = 50
C1  = 2
C2  = 2
OMEGA_MIN = 0.4
OMEGA_MAX = 0.7
MAX_VEL   = 0.7

#test_case_map = 
test_case_map = np.array([
    [0, 1, 4.0], [0, 2, 7.0], [0, 3, 3.0], [0, 4, 8.0],
    [1, 2, 5.0], [1, 3, 6.0], [1, 4, 9.0],
    [2, 3, 2.0], [2, 4, 6.0],
    [3, 4, 5.0],
]) 


if __name__ == "__main__":
    tsp = TSP(START_IDX, N_CITIES)
    tsp.build_map(test_case_map, False)

    tsp_manager = TSPManager(tsp, PUNISH_RATE)
    fitness_func = tsp_manager.get_target_func()
    pso_inst = MyPSO(
        n_max_iterations=N_MAX_ITERATION,
        n_particles=N_PARTICLES,
        n_dimensions=tsp.n_cities,
        f_target=fitness_func,
        c1=C1,
        c2=C2,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
        max_vel=MAX_VEL
    )
    pso_inst.pso_process()
    res = tsp_manager.from_float_score_to_int_path(pso_inst.g_best)
    print("最优路径访问顺序:", res)
    print("总长度:", tsp_manager.calculate_path_weight(res))
    print("断裂边数:", tsp_manager.check_path_connectivity(res))   # ← 关键诊断