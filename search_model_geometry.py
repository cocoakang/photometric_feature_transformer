import numpy as np
import random
from subprocess import Popen
import GPUtil
import time

if __name__ == "__main__":
    random.seed(2233)
    np.random.seed(3322)

    deviceIDs = GPUtil.getAvailable(limit = 5)
    gpu_num = len(deviceIDs)
    print("available gpu num:",gpu_num)
    tmp_seed = np.random.randint(0,9993748,size=5)
    task_counter = 0
    pool = []
    for m_len in [3,5,9,7,1]:
        for code_len in [7,5,9,3,128,64,32,16,11,1]:
            cur_gpu = deviceIDs[task_counter % gpu_num]
            tmp_process = Popen(
                [
                    "python",
                    "tame.py",
                    "../../training_data/aniso_log_40_2b_with_rotate/",
                    "--search_model",
                    "--training_gpu",
                    "{}".format(cur_gpu),
                    "--rendering_gpu",
                    "{}".format(cur_gpu),
                    "--checker_gpu",
                    "{}".format(cur_gpu),
                    "--start_seed",
                    "{}".format(8157),
                    "--torch_manual_seed",
                    "{}".format(182137),
                    "--torch_cuda_manual_seed_all",
                    "{}".format(182137),
                    "--train_mine_seed",
                    "{}".format(165491),
                    "--val_mine_seed",
                    "{}".format(992831),
                    "--m_len",
                    "{}".format(m_len),
                    "--code_len",
                    "{}".format(code_len),
                    "--search_which",
                    "geometry"
                ]
            )
            pool.append(tmp_process)
            task_counter+=1
            time.sleep(15.0)
            if len(pool) == 8:
                exit_codes = [p.wait() for p in pool]
                print("exit codes:",exit_codes)
                pool = []
                exit()
                