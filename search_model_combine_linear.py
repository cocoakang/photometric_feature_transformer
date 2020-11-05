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

    g_candidates = (
        (3,7,2),
        (3,9,5),
        (5,7,7)
    )
    m_candiates = (
        (3,7),
        (5,9),
        (7,9)
    )

    for m_m_len,m_d_len in m_candiates:
        for g_m_len,g_d_len,g_v in g_candidates:
            cur_gpu = deviceIDs[task_counter % gpu_num]
            tmp_process = Popen(
                [
                    "python",
                    "tame.py",
                    "../../training_data/aniso_log_40_2b_with_rotate/",
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
                    "--pretrained_model_pan_h",
                    "/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/model_trained/search_model_material/learn_l2_ml{}_mg0_dla0_dlna{}_dg0_h/models/model_state_90000.pkl".format(m_m_len,m_d_len),
                    "--pretrained_model_pan_v",
                    "/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/model_trained/search_model_geometry/learn_l2_ml0_mg{}_dla0_dlna0_dg{}_v{}/models/model_state_90000.pkl".format(g_m_len,g_d_len,g_v),

                ]
            )
            pool.append(tmp_process)
            task_counter+=1
            time.sleep(15.0)
            if len(pool) == 4:
                exit_codes = [p.wait() for p in pool]
                print("exit codes:",exit_codes)
                pool = []
                