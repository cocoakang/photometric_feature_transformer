import torch
import numpy as np
import math
import torch
import random
from multiprocessing import Process
from mine import Mine
from mine_hard import Mine_Hard
import time
import sys
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from multiview_renderer_mt import Multiview_Renderer
from torch_render import Setup_Config
from generate_training_data import compute_loss_weight

def run(args,name,setup,RENDER_SCALAR,output_queue,seed,noise_config):
    np.random.seed(seed)
    # torch.random.manual_seed(seed+1)
    # random.seed(seed+2)
    if noise_config is None:
        mine = Mine(args,name)
    else:
        print("not ready")
        exit()
        mine = Mine_Hard(args,name,noise_config)
    print("build mine done.")
    #######################################
    # define rendering module           ###
    #######################################
    
    print("[MINE PRO PROCESS] Starting...{}".format(name))
    while True:
        input_params,input_positions,input_frame,input_rt,input_rotate_angle = mine.generate_training_data()#(batchsize,((1+self.sample_view_num)*(7+3+3+3+3)+3)) torch_renderer
        batch_size = input_params.size()[0]//2
        device = input_params.device
   
        ####rendering input lumi
        rendered_result,end_points = torch_render.draw_rendering_net(
            setup,
            input_params,
            input_positions,
            input_rotate_angle,
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
        rendered_result = rendered_result.reshape(2,batch_size,setup.get_light_num(),1)

        training_data_map = {
            "input_lumi":rendered_result,
            "normal":normals_localview,
            "rt":input_rt
        }
        # print("[MINE PRO PROCESS] putting data...{}".format(self.mine.name))
        output_queue.put(training_data_map)
        # print("[MINE PRO PROCESS] done...{}".format(self.mine.name))
        


class Mine_Pro():
    def __init__(self,args,name,output_queue,output_sph,seed,noise_config=None):
        print("[MINE PRO {}] creating mine...".format(name))
        ##########
        ##parse arguments
        #########
        self.args = args
        self.name = name
        self.output_queue = output_queue
        self.setup = self.args["setup_input"]
        self.seed = seed
        self.noise_config = noise_config
        
        #######################################
        #loading setup configuration        ###
        #######################################
        
        self.RENDER_SCALAR = self.args["RENDER_SCALAR"]

        #######################################
        #loading projection kernels         ###
        #######################################

        print("[MINE PRO {}] creating mine pro done.".format(name))
    
    def start(self):
        self.generator = Process(target=run, args=(
            self.args,
            self.name,
            self.setup,
            self.RENDER_SCALAR,
            self.output_queue,
            self.seed,
            self.noise_config
        ),daemon=True)
        # self.generator.setDaemon(True)
        self.generator.start()