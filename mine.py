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

class Mine:
    def __init__(self,train_configs,name):
        np.random.seed(666)
        self.record_size = origin_param_dim*4
        self.batch_size = train_configs["batch_size"]
        self.buffer_size = train_configs["pre_load_buffer_size"]

        self.setup_input = train_configs["setup_input"]
        self.sample_view_num = train_configs["sample_view_num"]

        self.rendering_device = train_configs["rendering_device"]

        self.sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=self.sample_view_num,endpoint=False)#[self.sample_view_num]
        self.sampled_rotate_angles_np = np.expand_dims(self.sampled_rotate_angles_np,axis=0)
        self.sampled_rotate_angles_np = np.repeat(self.sampled_rotate_angles_np,self.batch_size,axis=0)
        self.sampled_rotate_angles_np = self.sampled_rotate_angles_np.astype(np.float32)
        self.sampled_rotate_angles_tc = torch.from_numpy(self.sampled_rotate_angles_np).to(self.rendering_device)
        self.sampled_rotate_angles_tc = self.sampled_rotate_angles_tc.reshape(self.batch_size*self.sample_view_num,1)#(batchsize*sample_view_num,1)

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

    def generate_training_data(self,test_tangent_flag = False):
        tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        if tmp_params.shape[0] < self.batch_size:
            self.__preload()
            tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        self.current_ptr+=self.batch_size

        tmp_params[:,6:8] = self.__rejection_sampling_axay(test_tangent_flag)
        # tmp_params[:,7] = tmp_params[:,6]
        new_positions = self.generate_batch_positions(self.batch_size)
        tmp_params = np.concatenate([
            tmp_params[:,3:3+5],
            self.sample_color(param_bounds["pd"][0],param_bounds["pd"][1]),
            self.sample_color(param_bounds["ps"][0],param_bounds["ps"][1])
        ],axis=-1)

        input_params = torch.from_numpy(tmp_params).to(self.rendering_device)
        input_positions = torch.from_numpy(new_positions[:,:3]).to(self.rendering_device)

        ##################################################################
        ###adjust position here to ensure at least two views are visible
        ##################################################################
        while True:
            n2d = input_params[:,:2]#(batch_size,2)
            # theta = tmp_param[:,[2]]#(batch_size,1)
            # view_dir = self.setup_input.get_cam_pos_torch(self.rendering_device) - input_positions #shape=[batch,3]
            # view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]

            # frame_t,frame_b = torch_render.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
            # frame_n = view_dir#[batch,3]

            # n_local = torch_render.back_hemi_octa_map(n2d)#[batch,3]
            # normal = n_local[:,[0]]*frame_t+n_local[:,[1]]*frame_b+n_local[:,[2]]*frame_n#[batch,3]
            normal = torch_render.back_full_octa_map(n2d)#(batch,3)

            tmp_normal = torch.unsqueeze(normal,dim=1).repeat(1,self.sample_view_num,1).reshape(self.batch_size*self.sample_view_num,3)
            tmp_position = torch.unsqueeze(input_positions,dim=1).repeat(1,self.sample_view_num,1).reshape(self.batch_size*self.sample_view_num,3)
            tmp_rotate_theta = self.sampled_rotate_angles_tc.clone()
            wo_dot_n = torch_render.compute_wo_dot_n(self.setup_input,tmp_position,tmp_rotate_theta,tmp_normal,self.setup_input.get_cam_pos_torch(self.rendering_device))#(remain*sampleviewnum,1)
            wo_dot_n = wo_dot_n.reshape(self.batch_size,self.sample_view_num)
            tmp_visible_flag = wo_dot_n > 0.0
            visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
            invalid_idxes = torch.where(visible_num < 2)[0]
            invalid_num = invalid_idxes.size()[0]
            if invalid_num == 0:
                break
            new_positions = torch.from_numpy(self.generate_batch_positions(invalid_num)).to(self.rendering_device)
            new_n2d = torch.from_numpy(np.random.uniform(param_bounds["n"][0],param_bounds["n"][1],[invalid_num,2]).astype(np.float32)).to(self.rendering_device)
            input_positions[invalid_idxes] = new_positions
            input_params[invalid_idxes,:2] = new_n2d
        ##################################################################
        ###select two visible view
        ##################################################################
        choosed_idx_x = np.stack([np.array(range(self.batch_size)),np.array(range(self.batch_size))],axis=1).reshape([-1])
        mask_matrix = tmp_visible_flag.cpu().numpy()
        try:
            choosed_idx = np.stack([np.random.choice(np.where(mask_matrix[i])[0],size=2,replace=False) for i in range(self.batch_size)]).reshape([-1])
        except BaseException as e:
            print(e)
            input_params.cpu().numpy().astype(np.float32).tofile("error_param.bin")
            input_positions.cpu().numpy().astype(np.float32).tofile("error_pos.bin")
            exit()

        chossed_roate_angles = np.reshape(self.sampled_rotate_angles_np[choosed_idx_x,choosed_idx],[self.batch_size,2])
        chossed_roate_angles = torch.from_numpy(chossed_roate_angles).to(self.rendering_device)

        return input_params,input_positions,chossed_roate_angles

    def generate_validating_data(self,test_tangent_flag):
        return self.generate_training_data(test_tangent_flag)