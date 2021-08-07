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
param_bounds["box"] = (-50.0,50.0)
param_bounds["angle"] = (0.0,2.0*math.pi)

class Mine:
    def __init__(self,train_configs,name):
        # np.random.seed(666)
        self.record_size = origin_param_dim*4
        self.batch_size = train_configs["batch_size"]
        # self.batch_brdf_num = train_configs["batch_brdf_num"]
        self.buffer_size = train_configs["pre_load_buffer_size"]
        # assert self.batch_brdf_num*2 <= self.batch_size,"batch_brdf_num:{} batch_size:{}".format(self.batch_brdf_num,self.batch_size)

        self.setup_input = train_configs["setup_input"]
        # self.sample_view_num = train_configs["sample_view_num"]

        self.rendering_device = train_configs["rendering_device"]

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

    def generate_batch_positions_incf(self,batch_size,bounding="box"):
        tmp_pos = np.random.uniform(param_bounds[bounding][0],param_bounds[bounding][1],[batch_size,3]).astype(np.float32)
        tmp_pos[:,2] = tmp_pos[:,2] + 500
        return tmp_pos
    
    def generate_batch_positions(self,batch_size,bounding="box"):
        return np.random.uniform(param_bounds[bounding][0],param_bounds[bounding][1],[batch_size,3]).astype(np.float32)

    def generate_batch_frame(self,batch_size):
        '''
        positions_1, positions_2: numpy array, camera frame
        '''
        positions = torch.from_numpy(self.generate_batch_positions(batch_size).astype(np.float32)).to(self.rendering_device)#global space
        n_2d = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(self.rendering_device)
        rotate_theta = torch.from_numpy(np.random.uniform(0.0,math.pi,(batch_size,2)).astype(np.float32)).to(self.rendering_device)#(batch_size,2) CAUTION! rotate obeject with axis, if apply to camera, it should be inversed!

        ##################################################################
        ###adjust position here to ensure at least two views are visible
        ##################################################################
        while True:
            n2d = n_2d

            normal = torch_render.back_full_octa_map(n2d)#(batch,3) normal global frame
            tmp_normal = normal.reshape((batch_size,1,3,1)).repeat(1,2,1,1).reshape(batch_size*2,3)#(batch_size*2,3)
            
            tmp_position = positions.reshape(batch_size,1,3,1).repeat(1,2,1,1).reshape(batch_size*2,3)#(batch_size*2,3)

            tmp_rotate_theta = rotate_theta.reshape(batch_size*2,1)#(batch_size*2,1)

            wo_dot_n = torch_render.compute_wo_dot_n(self.setup_input,tmp_position,tmp_rotate_theta,tmp_normal,self.setup_input.get_cam_pos_torch(self.rendering_device))#(batch_size*2,1)
            
            wo_dot_n = wo_dot_n.reshape(self.batch_size,2)
            tmp_visible_flag = wo_dot_n > 0.0
            visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
            invalid_idxes = torch.where(visible_num < 2)[0]
            invalid_num = invalid_idxes.size()[0]
            if invalid_num == 0:
                break

            new_positions = torch.from_numpy(self.generate_batch_positions(invalid_num)).to(self.rendering_device)
            new_n2d = torch.from_numpy(np.random.rand(invalid_num,2).astype(np.float32)).to(self.rendering_device)
            new_rotate_theta = torch.from_numpy(np.random.uniform(0.0,math.pi,(invalid_num,2)).astype(np.float32)).to(self.rendering_device)
            positions[invalid_idxes] = new_positions
            n_2d[invalid_idxes] = new_n2d
            rotate_theta[invalid_idxes] = new_rotate_theta
        
        #########################################


        n_global = torch_render.back_full_octa_map(n_2d)#(batch,3) normal global
        theta = torch.from_numpy(np.random.rand(batch_size,1).astype(np.float32)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]).to(self.rendering_device)
        t_global,_ = torch_render.build_frame_f_z(n_global,theta,with_theta=True)
        b_global = torch.cross(n_global,t_global)

        frame_global = [n_global,t_global,b_global]#(#(n,t,b) n is (batchsize,3))

        rotated_R_matrix,rotated_T_vec = self.setup_input.get_rts(-rotate_theta.reshape((self.batch_size*2,1)),self.rendering_device)

        rotated_R_matrix = rotated_R_matrix.reshape((self.batch_size,2,3,3))
        R_matrix_1,R_matrix_2 = rotated_R_matrix[:,0],rotated_R_matrix[:,1]
        rotated_T_vec = rotated_T_vec.reshape((self.batch_size,2,3,1))
        T_vec_1,T_vec_2 = rotated_T_vec[:,0],rotated_T_vec[:,1]

        return frame_global,positions,R_matrix_1,R_matrix_2,T_vec_1,T_vec_2,rotate_theta


    def generate_training_data(self,test_tangent_flag = False):
        tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        if tmp_params.shape[0] < self.batch_size:
            self.__preload()
            tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        self.current_ptr+=self.batch_size

        tmp_params[:,6:8] = self.__rejection_sampling_axay(test_tangent_flag)
        
        frame_global,positions_global,R_matrix_1,R_matrix_2,t_vec_1,t_vec_2,rotate_theta = self.generate_batch_frame(self.batch_size)
     
        rt_1 = torch.cat((
                R_matrix_1.reshape((-1,3*3)),
                t_vec_1.reshape((-1,3))
            ),dim=1
        )#(batchsize,12))

        rt_2 = torch.cat((
                R_matrix_2.reshape((-1,3*3)),
                t_vec_2.reshape((-1,3))
            ),dim=1
        )#(batchsize,12)

        tmp_params = tmp_params[:,3:3+7]

        input_params = torch.from_numpy(tmp_params).to(self.rendering_device)#(batchsize,param_dim)
        input_positions = positions_global#(batchsize,3)
        input_rotate_angle = rotate_theta.to(self.rendering_device)#(batchsize,2)
        input_frame = frame_global#[(batchsize,3),(batchsize,3),(batchsize,3)]

        return input_params,input_positions,input_rotate_angle,input_frame,rt_1,rt_2

    def generate_validating_data(self,test_tangent_flag):
        return self.generate_training_data(test_tangent_flag)