import torch
import numpy as np
import math
import torch
import random
from multiprocessing import Process
from mine import Mine
import time
import sys
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from multiview_renderer_mt import Multiview_Renderer
from generate_training_data import compute_loss_weight

def run(args,name,setup,output_queue,seed):
    np.random.seed(seed)
    RENDER_SCALAR = args["RENDER_SCALAR"]
    # torch.random.manual_seed(seed+1)
    # random.seed(seed+2)
    mine = Mine(args,name)
    print("build mine done.")
    #######################################
    # define rendering module           ###
    #######################################

    print("[MINE PRO PROCESS] Starting...{}".format(name))
    while True:
        input_params,input_positions,input_rotate_angle,input_frame,rt_1,rt_2 = mine.generate_training_data()
        batch_size = input_params.size()[0]
        device = input_params.device
   
        origin_param_dim = input_params.shape[1]
        ####rendering input lumi
        input_frame = [tmp_axis.unsqueeze(dim=1).repeat(1,2,1).reshape((batch_size*2,3)) for tmp_axis in input_frame]
        rendered_result,end_points = torch_render.draw_rendering_net(
            setup,
            input_params.unsqueeze(dim=1).repeat(1,2,1).reshape((batch_size*2,origin_param_dim)),
            input_positions.unsqueeze(dim=1).repeat(1,2,1).reshape((batch_size*2,3)),
            input_rotate_angle.reshape((batch_size*2,1)),
            "rendering_input_slice",
            global_custom_frame=input_frame,
            use_custom_frame="ntb"
        )
        
        n_dot_wo = end_points["n_dot_view_dir"]#(2*batch,1)
        normals_localview = end_points["n"]#(2*batch,3)
        # normals_localview = torch.reshape(normals_localview,(batch_size,2,3)).permute(1,0,2).reshape(2*batch_size,3)
        visibility = (n_dot_wo.reshape(-1) > 0.0).all()
        # print(visibility)
        if not visibility:
            print(n_dot_wo)
            print("error occured!")
            exit()

        rendered_result = rendered_result*RENDER_SCALAR
        rendered_result = rendered_result.reshape(batch_size,2,setup.get_light_num(),1)

        training_data_map = {
            "input_lumi":rendered_result,
            "normal":None,
            "position":None,
            "rts":[rt_1,rt_2]
        }
        # print("[MINE PRO PROCESS] putting data...{}".format(name))
        output_queue.put(training_data_map)
        # print("[MINE PRO PROCESS] done...{}".format(name))
        


class Mine_Pro():
    def __init__(self,args,name,output_queue,seed):
        print("[MINE PRO {}] creating mine...".format(name))
        ##########
        ##parse arguments
        #########
        self.args = args
        self.name = name
        self.output_queue = output_queue
        self.setup = self.args["setup_input"]
        self.seed = seed
        
        #######################################
        #loading setup configuration        ###
        #######################################
        

        #######################################
        #loading projection kernels         ###
        #######################################

        print("[MINE PRO {}] creating mine pro done.".format(name))
    
    def start(self):
        self.generator = Process(target=run, args=(
            self.args,
            self.name,
            self.setup,
            self.output_queue,
            self.seed
        ),daemon=True)
        # self.generator.setDaemon(True)
        self.generator.start()