import torch
import numpy as np
import math
import torch
import random
import threading
from mine import Mine
import time
import sys
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from multiview_renderer_mt import Multiview_Renderer
from torch_render import Setup_Config
from generate_training_data import compute_loss_weight

def run(args,name,setup,RENDER_SCALAR,output_queue):
    # np.random.seed(23333)
    # torch.random.manual_seed(23333)
    # random.seed(666)
    mine = Mine(args,name)
    print("build mine done.")
    #######################################
    # define rendering module           ###
    #######################################
    
    print("[MINE PRO PROCESS] Starting...{}".format(name))
    while True:
        # print("[MINE PRO PROCESS] get data...{}".format(self.mine.name))
        param,position,rotate_angles = mine.generate_training_data()#(batchsize,((1+self.sample_view_num)*(7+3+3+3+3)+3)) torch_renderer
        # print("[MINE PRO PROCESS] get data done...{}".format(self.mine.name))
        batch_size = param.size()[0]
        device = param.device
   
        ### build frame
        n2d = param[:,:2]#(batch_size,2)
        theta = param[:,[2]]#(batch_size,1)

        # view_dir = args["setup_input"].get_cam_pos_torch(position.device) - position #shape=[batch,3]
        # view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]
        # frame_t,frame_b = torch_render.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        # frame_n = view_dir#[batch,3]

        # n_local = torch_render.back_hemi_octa_map(n2d)#[batch,3]
        # t_local,_ = torch_render.build_frame_f_z(n_local,theta,with_theta=True)
        # n = n_local[:,[0]]*frame_t+n_local[:,[1]]*frame_b+n_local[:,[2]]*frame_n#[batch,3]
        # t = t_local[:,[0]]*frame_t+t_local[:,[1]]*frame_b+t_local[:,[2]]*frame_n#[batch,3]
        # b = torch.cross(n,t)#[batch,3]
        
        normal = torch_render.back_full_octa_map(n2d)#(batch_size,3)
        tangent,binormal =torch_render.build_frame_f_z(normal,theta)
        global_frame = [normal,tangent,binormal]
        # global_frame=[n,t,b]

        ###double param 
        param_2 = torch.unsqueeze(param,dim=1).repeat(1,2,1).reshape(batch_size*2,7)
        position_2 = torch.unsqueeze(position,dim=1).repeat(1,2,1).reshape(batch_size*2,3)
        for i in range(3):
            global_frame[i] = torch.unsqueeze(global_frame[i],dim=1).repeat(1,2,1).reshape(batch_size*2,3)
        rotate_angles = rotate_angles.reshape(batch_size*2,1)
        ### rendering input lumi
        rendered_result,end_points = torch_render.draw_rendering_net(
            setup,
            param_2,
            position_2,
            rotate_angles,
            "rendering_input_slice",
            global_custom_frame=global_frame,
            use_custom_frame="ntb"
        )
        
        n_dot_wo = end_points["n_dot_view_dir"]#(batch*2,1)
        normals_localview = end_points["n"]#(batch*2,3)
        normals_localview = torch.reshape(normals_localview,(batch_size,2,3)).permute(1,0,2).reshape(2*batch_size,3)
        normals_globalview = torch.unsqueeze(normal,dim=0).repeat(2,1,1).reshape(2*batch_size,3)#(2*batch,3)
        visibility = torch.where(~(n_dot_wo.reshape(batch_size,2) > 0.0).all(dim=1))[0]
        # print(visibility)
        if len(visibility) > 0:
            print("error occured!")
            exit()

        rendered_result = rendered_result*RENDER_SCALAR
        rendered_result = rendered_result.reshape(batch_size,2,setup.get_light_num(),1)
        rendered_result = rendered_result.permute(1,0,2,3)

        # print("[MINE PRO PROCESS] contructing...{}".format(self.mine.name))
        # print("param {}:".format(name),param[-3:])
        # print("position {}:".format(name),position[-3:])
        # exit()

        rotate_angles = rotate_angles.reshape(batch_size,2,1).permute(1,0,2).reshape(2*batch_size,1)
        view_ids_cossin = torch.cat(
            [
                torch.sin(rotate_angles),
                torch.cos(rotate_angles)
            ],dim=1
        )
        position_2 = position_2.reshape(batch_size,2,3).permute(1,0,2).reshape(2*batch_size,3)
        # position_2 = end_points["position"].reshape(batch_size,2,3)
        # position_2 = position_2.permute(1,0,2).reshape(2*batch_size,3)

        training_data_map = {
            "input_lumi":rendered_result,
            "param":param,
            "position":position,
            "position_2":position_2,
            "normal_local":normals_localview,
            "normal":normals_globalview,
            "view_ids_cossin":view_ids_cossin,
            "rotate_theta":rotate_angles
        }
        # print("[MINE PRO PROCESS] putting data...{}".format(self.mine.name))
        output_queue.put(training_data_map)
        # print("[MINE PRO PROCESS] done...{}".format(self.mine.name))
        


class Mine_Pro():
    def __init__(self,args,name,output_queue,output_sph):
        print("[MINE PRO {}] creating mine...".format(name))
        ##########
        ##parse arguments
        #########
        self.args = args
        self.name = name
        self.output_queue = output_queue
        self.setup = self.args["setup_input"]
        
        #######################################
        #loading setup configuration        ###
        #######################################
        
        self.RENDER_SCALAR = 5*1e3/math.pi

        #######################################
        #loading projection kernels         ###
        #######################################

        print("[MINE PRO {}] creating mine pro done.".format(name))
    
    def start(self):
        self.generator = threading.Thread(target=run, args=(
            self.args,
            self.name,
            self.setup,
            self.RENDER_SCALAR,
            self.output_queue
        ))
        self.generator.setDaemon(True)
        self.generator.start()