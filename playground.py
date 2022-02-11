import re

txt = "/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/model_trained/search_model_material/learn_l2_ml3_mg0_dla0_dlna7_dg0_h/models/model_state_90000.pkl"
x = re.search("_dlna", txt)
start_idx = x.start()+len("_dlna")
m_len = int(txt[start_idx:start_idx+1])
print(m_len)