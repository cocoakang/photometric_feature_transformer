import numpy as np
import math
import sys
sys.path.append("../torch_renderer/")
import torch_render
import torch

origin_param_dim = 7+3+1
epsilon = 1e-3
param_bounds={}
param_bounds["n"] = (epsilon,1.0-epsilon)
param_bounds["theta"] = (0.0,math.pi)
param_bounds["a"] = (0.006,0.503)
param_bounds["pd"] = (0.0,1.0)
param_bounds["ps"] = (0.0,10.0)
param_bounds["box"] = (-75.0,75.0)
param_bounds["angle"] = (0.0,2.0*math.pi)

class Mine_Hard:
    def __init__(self,train_configs,name,noise_config):
        # np.random.seed(666)
        self.record_size = origin_param_dim*4
        self.batch_size = train_configs["batch_size"]
        # self.batch_brdf_num = train_configs["batch_brdf_num"]
        self.buffer_size = train_configs["pre_load_buffer_size"]
        # assert self.batch_brdf_num*2 <= self.batch_size,"batch_brdf_num:{} batch_size:{}".format(self.batch_brdf_num,self.batch_size)

        self.setup_input = train_configs["setup_input"]
        # self.sample_view_num = train_configs["sample_view_num"]

        self.rendering_device = train_configs["rendering_device"]
        self.noise_config = noise_config

        # self.sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=self.sample_view_num,endpoint=False)#[self.sample_view_num]
        # self.sampled_rotate_angles_np = np.expand_dims(self.sampled_rotate_angles_np,axis=0)
        # self.sampled_rotate_angles_np = np.repeat(self.sampled_rotate_angles_np,self.batch_size,axis=0)
        # self.sampled_rotate_angles_np = self.sampled_rotate_angles_np.astype(np.float32)
        # self.sampled_rotate_angles_tc = torch.from_numpy(self.sampled_rotate_angles_np).to(self.rendering_device)
        # self.sampled_rotate_angles_tc = self.sampled_rotate_angles_tc.reshape(self.batch_size*self.sample_view_num,1)#(batchsize*sample_view_num,1)

        self.name = name
        if self.name == "train":
            self.pf_train = open(train_configs["data_root"]+"cooked_params_train.bin","rb")
        elif self.name == "val":
            self.pf_train = open(train_configs["data_root"]+"cooked_params_val.bin","rb")
            self.buffer_size = 100000

        self.pf_train.seek(0,2)
        self.train_data_size = self.pf_train.tell()//4//origin_param_dim

        assert self.train_data_size > 0

        print("[MINE]"+self.name+" train data size:",self.train_data_size)

        self.available_train_idx = list(range(self.train_data_size))

        self.train_ptr = 0

        self.__refresh_train()
        self.__preload()

    def __preload(self):
        print(self.name+" preloading...")
        tmp_params_idxes = self.available_train_idx[self.train_ptr:self.train_ptr+self.buffer_size]
        if len(tmp_params_idxes) == 0:
            self.__refresh_train()
            tmp_params_idxes = self.available_train_idx[self.train_ptr:self.train_ptr+self.buffer_size]
        self.train_ptr+=len(tmp_params_idxes)

        tmp_map = {tmp_params_idxes[i]:i for i in range(len(tmp_params_idxes))}

        tmp_params_idxes.sort()

        self.buffer_params = np.zeros([len(tmp_params_idxes),origin_param_dim],np.float32)
        for idx in range(len(tmp_params_idxes)):
            self.pf_train.seek(tmp_params_idxes[idx]*self.record_size,0)
            self.buffer_params[tmp_map[tmp_params_idxes[idx]]] = np.fromfile(self.pf_train,np.float32,origin_param_dim)
        
        self.current_ptr = 0

        print(self.name+" done.")

    def __refresh_train(self):
        print(self.name+" refreshing train...")
        np.random.shuffle(self.available_train_idx)
        self.train_ptr = 0
        print(self.name+" done.")

    def __rejection_sampling_axay(self,test_tangent_flag):
        origin = np.exp(np.random.uniform(np.log(param_bounds["a"][0]),np.log(param_bounds["a"][1]),[self.batch_size,2]))
        # origin = np.where(origin[:,[0]]>origin[:,[1]],origin,origin[:,::-1])
        while True:
            still_need = np.logical_and(origin[:,0]>0.35,origin[:,1] >0.35)
            # if test_tangent_flag:
            #     still_need = np.logical_or(still_need,origin[:,1]*3>origin[:,0])
            where = np.nonzero(still_need)
            num = where[0].shape[0]
            if num == 0:
                break
            new_data= np.exp(np.random.uniform(np.log(param_bounds["a"][0]),np.log(param_bounds["a"][1]),[num,2]))
            # new_data = np.where(new_data[:,[0]]>new_data[:,[1]],new_data,new_data[:,::-1])
            origin[where] = new_data
        return origin

    def sample_color(self,min_rho,max_rho):
        '''
        return [batch_size,3]
        '''
        tmp_color = np.random.rand(self.batch_size,3)*(max_rho-min_rho) + min_rho
        return tmp_color.astype(np.float32)

    def generate_batch_positions(self,batch_size):
        return np.concatenate([
            np.random.uniform(param_bounds["box"][0],param_bounds["box"][1],[batch_size,2]),
            np.random.uniform(-30.0,120.0,[batch_size,1])
        ],axis=-1).astype(np.float32)
    
    def gen_position_noise(self,batch_size):
        #TODO add anisotropic here
        tmp_noise = np.random.randn(batch_size,3)*self.noise_config["position"]
        tmp_noise = torch.from_numpy(tmp_noise.astype(np.float32)).to(self.rendering_device)
        return tmp_noise
    
    def gen_frame_noise_uniform(self,batch_size):
        tmp_theta_h = torch.from_numpy(np.random.uniform(-math.pi*0.1,math.pi*0.1,size=(batch_size,1)).astype(np.float32)).to(self.rendering_device)
        tmp_theta_v = torch.from_numpy(np.random.uniform(-math.pi*0.1,math.pi*0.1,size=(batch_size,1)).astype(np.float32)).to(self.rendering_device)
        
        tmp_frame_noise = torch.cat([tmp_theta_h,tmp_theta_v],dim=1)
        
        return tmp_frame_noise

    def gen_frame_noise_gaussian(self,batch_size):
        tmp_theta_h = np.random.randn(batch_size,1)*self.noise_config["frame_normal_h"]/180.0*math.pi
        tmp_theta_h = torch.from_numpy(tmp_theta_h.astype(np.float32)).to(self.rendering_device)


        tmp_theta_v = np.random.randn(batch_size,1)*self.noise_config["frame_normal_v"]/180.0*math.pi
        tmp_theta_v = torch.from_numpy(tmp_theta_v.astype(np.float32)).to(self.rendering_device)
        
        tmp_frame_noise = torch.cat([tmp_theta_h,tmp_theta_v],dim=1)
        
        return tmp_frame_noise
    
    def gen_theta_noise(self,batch_size):
        tmp_noise = np.random.randn(batch_size,1)*self.noise_config["theta"]+1.0
        tmp_noise = torch.from_numpy(tmp_noise.astype(np.float32)).to(self.rendering_device)
        return tmp_noise
    
    def gen_axay_noise(self,batch_size):
        tmp_noise = np.random.randn(batch_size,1)*self.noise_config["axay"]+1.0
        tmp_noise = torch.from_numpy(tmp_noise.astype(np.float32)).to(self.rendering_device)
        return tmp_noise
    
    def gen_pd_noise(self,batch_size):
        tmp_noise = np.random.randn(batch_size,1)*self.noise_config["pd"]+1.0
        tmp_noise = torch.from_numpy(tmp_noise.astype(np.float32)).to(self.rendering_device)
        return tmp_noise
    
    def gen_ps_noise(self,batch_size):
        tmp_noise = np.random.randn(batch_size,1)*self.noise_config["ps"]+1.0
        tmp_noise = torch.from_numpy(tmp_noise.astype(np.float32)).to(self.rendering_device)
        return tmp_noise

    def disturb_frame(self,normal,frame_noise):
        '''
        normal = (batchsize,3)
        frame_noise = (batchsize,2) h_angle,v_angle(radian656
        '''
        #about h
        origin_theta_xy = torch.nn.functional.normalize(normal[:,:2],dim=1)
        origin_theta = torch.atan2(origin_theta_xy[:,1],origin_theta_xy[:,0])#(batchsize,)
        disturbed_theta =origin_theta+frame_noise[:,0]
        disturbed_theta_xy = torch.stack([torch.sin(disturbed_theta),torch.cos(disturbed_theta)],dim=1)#(batchsize,2)

        #about v
        origin_phi = torch.asin(normal[:,2])#(batchsize,)
        disturbed_phi =  torch.clamp(origin_phi+frame_noise[:,1],-math.pi*0.5,math.pi*0.5)#(batchsize,)
        disturbed_z = torch.sin(disturbed_phi)#(batchsize,)
        #compute new xy
        disturbed_normal_xylen = torch.abs(torch.cos(disturbed_phi))#(batchsize,)
        disturbed_xy = disturbed_theta_xy*disturbed_normal_xylen.unsqueeze(dim=1)

        disturbed_normal = torch.cat([disturbed_xy,disturbed_z.unsqueeze(dim=1)],dim=1)#(batchsize,3)

        disturbed_normal = torch.nn.functional.normalize(disturbed_normal,dim=1)

        return disturbed_normal

    def generate_training_data(self,test_tangent_flag = False):
        tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        if tmp_params.shape[0] < self.batch_size:
            self.__preload()
            tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        self.current_ptr+=self.batch_size

        tmp_params[:,6:8] = self.__rejection_sampling_axay(test_tangent_flag)
        # tmp_params[:,7] = tmp_params[:,6]
        new_positions = self.generate_batch_positions(self.batch_size)
        tmp_params = tmp_params[:,3:3+7]

        recompute_needed = True

        while True:
            base_case_id = np.random.randint(0,self.batch_size)

            input_params = torch.from_numpy(tmp_params).to(self.rendering_device)
            input_positions = torch.from_numpy(new_positions[:,:3]).to(self.rendering_device)

            #to let all share same params, noise will be added
            input_params[:] = input_params[base_case_id]
            input_positions[:] = input_positions[base_case_id]
            position_noise = self.gen_position_noise(self.batch_size)#(batchsize,3)
            frame_noise = self.gen_frame_noise_gaussian(self.batch_size)#(batchsize,2) h_angle,v_angle(radians)

            chossed_roate_angles = np.random.uniform(0.0,math.pi*2.0,[1,2]).astype(np.float32)
            chossed_roate_angles = np.repeat(chossed_roate_angles,self.batch_size,axis=0)
            chossed_roate_angles = torch.from_numpy(chossed_roate_angles).to(self.rendering_device)

            while True:
                n2d = input_params[:,:2]#(batch_size,2)
                normal = torch_render.back_full_octa_map(n2d)#(batch,3)
                
                tmp_normal = torch.unsqueeze(normal,dim=1).repeat(1,2,1).reshape(self.batch_size*2,3)
                tmp_position = torch.unsqueeze(input_positions + position_noise,dim=1).repeat(1,2,1).reshape(self.batch_size*2,3)
                tmp_rotate_theta = chossed_roate_angles.clone().reshape(self.batch_size*2,1)
                wo_dot_n = torch_render.compute_wo_dot_n(self.setup_input,tmp_position,tmp_rotate_theta,tmp_normal,self.setup_input.get_cam_pos_torch(self.rendering_device))#(remain*sampleviewnum,1)
                wo_dot_n = wo_dot_n.reshape(self.batch_size,2)
                tmp_visible_flag = wo_dot_n > 0.0
                visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
                invalid_idxes = torch.where(visible_num < 2)[0]
                invalid_num = invalid_idxes.size()[0]
                if invalid_num == 0:
                    break
                input_positions[:] = torch.from_numpy(self.generate_batch_positions(1)).to(self.rendering_device)#(1,3)
                input_params[:,:2] = torch.from_numpy(np.random.rand(2).astype(np.float32)).to(self.rendering_device)

            ##################################################################
            ###adjust position here to ensure at least two views are visible
            ##################################################################
            itr_counter = 0
            while True:
                n2d = input_params[:,:2]#(batch_size,2)
                normal = torch_render.back_full_octa_map(n2d)#(batch,3)
                # tangent,binormal = torch_render.build_frame_f_z(normal,None,with_theta=False)

                # disturb_dir = tangent*frame_noise[:,[0]]+binormal*frame_noise[:,[1]]
                # disturbed_normal = normal+disturb_dir*frame_noise[:,[2]]
                disturbed_normal = self.disturb_frame(normal,frame_noise)
                n2d_disturbed = torch_render.full_octa_map(disturbed_normal)
                disturbed_normal = torch_render.back_full_octa_map(n2d_disturbed)#to deal with float accuracy problem

                tmp_normal = torch.unsqueeze(disturbed_normal,dim=1).repeat(1,2,1).reshape(self.batch_size*2,3)
                tmp_position = torch.unsqueeze(input_positions + position_noise,dim=1).repeat(1,2,1).reshape(self.batch_size*2,3)
                tmp_rotate_theta = chossed_roate_angles.clone().reshape(self.batch_size*2,1)
                wo_dot_n = torch_render.compute_wo_dot_n(self.setup_input,tmp_position,tmp_rotate_theta,tmp_normal,self.setup_input.get_cam_pos_torch(self.rendering_device))#(remain*sampleviewnum,1)
                wo_dot_n = wo_dot_n.reshape(self.batch_size,2)
                tmp_visible_flag = wo_dot_n > 1e-5
                visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
                invalid_idxes = torch.where(visible_num < 2)[0]
                invalid_num = invalid_idxes.size()[0]
                # print("invalid_num:",invalid_num," invalid_idxes:",invalid_idxes)
                if invalid_num == 0:
                    recompute_needed = False
                    break
                elif itr_counter > 10:
                    break
                new_position_noise = self.gen_position_noise(invalid_num)#(invalidnum,3)
                new_frame_noise = self.gen_frame_noise_gaussian(invalid_num)
                position_noise[invalid_idxes] = new_position_noise
                frame_noise[invalid_idxes] = new_frame_noise

                itr_counter+=1

            if not recompute_needed:
                break

        input_positions = input_positions + position_noise
        input_params = torch.cat([
            n2d_disturbed,
            input_params[:,[2]]*self.gen_theta_noise(self.batch_size),
            torch.clamp(input_params[:,3:5]*self.gen_axay_noise(self.batch_size),param_bounds["a"][0],param_bounds["a"][1]),
            torch.clamp(input_params[:,[5]]*self.gen_pd_noise(self.batch_size),param_bounds["pd"][0],param_bounds["pd"][1]),
            torch.clamp(input_params[:,[6]]*self.gen_ps_noise(self.batch_size),param_bounds["ps"][0],param_bounds["ps"][1])
        ],dim=1)

        return input_params,input_positions,chossed_roate_angles

    def generate_validating_data(self,test_tangent_flag):
        return self.generate_training_data(test_tangent_flag)