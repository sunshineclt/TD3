import os
import multiprocessing as mp


def task(condition):
    algorithm, seed, actor_lr = condition
    env = "Ant-v1"
    if env == "Ant-v1" or env == "HalfCheetah-v1":
        start_timesteps = 10000
    else:
        start_timesteps = 1000
    command = "python main.py "
    command += "--env_name=" + env + " "
    command += "--policy_name=" + algorithm + " "
    command += "--seed=" + str(seed) + " "
    command += "--start_timesteps=" + str(start_timesteps) + " "
    command += "--actor_lr=" + str(actor_lr) + " "
    command += "--save_models "
    command += "--is_ro "
    os.system(command)


pool = mp.Pool(processes=25)

conditions = []
for algorithm in ["TD3"]:
    for seed in range(5):
        for actor_lr in [1e-3, 3e-4, 1e-4]:
            conditions.append((algorithm, seed, actor_lr))

pool.map(task, conditions)
