import numpy as np
import torch

class Mine_Sphere:
    def __init__(self,configs):
        self.x_sample_num = configs["x_sample_num"]
        self.y_sample_num = configs["y_sample_num"]
        
    def get_batch_data(self):
