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
param_bounds["box"] = (-100.0,100.0)
param_bounds["box_global"] = (-100.0,100.0)
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

        self.rts = self.setup_input.get_all_rts(self.rendering_device)#(list of (R_matrix,T_vec))

        self.R_matrixs = []
        self.T_vecs = []
        for R_matrix,T_vec in self.rts:
            self.R_matrixs.append(R_matrix)
            self.T_vecs.append(T_vec)

        self.R_matrixs = torch.stack(self.R_matrixs,dim=0)
        self.T_vecs = torch.stack(self.T_vecs,dim=0)

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

    def generate_batch_positions_incf(self,batch_size,bounding="box"):
        tmp_pos = np.random.uniform(param_bounds[bounding][0],param_bounds[bounding][1],[batch_size,3]).astype(np.float32)
        tmp_pos[:,2] = tmp_pos[:,2] + 1500
        return tmp_pos
    
    def generate_batch_positions(self,batch_size,bounding="box"):
        return np.random.uniform(param_bounds[bounding][0],param_bounds[bounding][1],[batch_size,3]).astype(np.float32)

    def generate_batch_visible_frame_incf(self,position):
        bath_size = position.shape[0]

        position = torch.from_numpy(position).to(self.rendering_device)
        n_2d = torch.from_numpy(np.random.rand(bath_size,2).astype(np.float32)*(param_bounds["n"][1]-param_bounds["n"][0])+param_bounds["n"][0]).to(self.rendering_device)
        theta = torch.from_numpy(np.random.rand(bath_size,1).astype(np.float32)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]).to(self.rendering_device)

        view_dir = torch.zeros_like(self.setup_input.get_cam_pos_torch(self.rendering_device)) - position #shape=[batch,3]
        view_dir = torch.nn.functional.normalize(view_dir,dim=1)#shape=[batch,3]
        #build local frame
        frame_t,frame_b = torch_render.build_frame_f_z(view_dir,None,with_theta=False)#[batch,3]
        frame_n = view_dir#[batch,3]

        n_local = torch_render.back_hemi_octa_map(n_2d)#[batch,3]
        t_local,_ = torch_render.build_frame_f_z(n_local,theta,with_theta=True)
        n = n_local[:,[0]]*frame_t+n_local[:,[1]]*frame_b+n_local[:,[2]]*frame_n#[batch,3]
        t = t_local[:,[0]]*frame_t+t_local[:,[1]]*frame_b+t_local[:,[2]]*frame_n#[batch,3]
        b = torch.cross(n,t)#[batch,3]

        return [n,t,b]

    def generate_batch_frame(self,batch_size):
        '''
        positions_1, positions_2: numpy array, camera frame
        '''
        positions = torch.from_numpy(self.generate_batch_positions(batch_size).astype(np.float32)).to(self.rendering_device)#global space
        n_2d = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(self.rendering_device)

        available_rt_num = self.R_matrixs.shape[0]
        chosen_rt_id = np.random.rand(batch_size,available_rt_num).argsort(axis=1)[:,:2]#(batch_size,2)

        ##################################################################
        ###adjust position here to ensure at least two views are visible
        ##################################################################
        while True:
            n2d = n_2d

            tmp_R_matrixs = self.R_matrixs[chosen_rt_id.reshape(-1)].reshape((batch_size,2,3,3))
            tmp_T_vecs = self.T_vecs[chosen_rt_id.reshape((-1))].reshape((batch_size,2,3,1))

            normal = torch_render.back_full_octa_map(n2d)#(batch,3) normal global frame
            normal = normal.reshape((batch_size,1,3,1)).repeat(1,2,1,1)#(batch_size,2,3,1)
            tmp_normal = torch.matmul(tmp_R_matrixs,normal).reshape(batch_size*2,3)#(batch_size*2,3) local frame

            tmp_position = positions.reshape(batch_size,1,3,1).repeat(1,2,1,1)
            tmp_position = (torch.matmul(tmp_R_matrixs,tmp_position)+tmp_T_vecs).reshape(batch_size*2,3)

            tmp_rotate_theta = torch.zeros(batch_size*2,1,dtype=torch.float32,device=self.rendering_device)

            wo_dot_n = torch_render.compute_wo_dot_n(self.setup_input,tmp_position,tmp_rotate_theta,tmp_normal,self.setup_input.get_cam_pos_torch(self.rendering_device))#(remain*sampleviewnum,1)
            
            wo_dot_n = wo_dot_n.reshape(self.batch_size,2)
            tmp_visible_flag = wo_dot_n > 0.0
            visible_num = torch.sum(torch.where(tmp_visible_flag,torch.ones_like(wo_dot_n),torch.zeros_like(wo_dot_n)),dim=1)
            invalid_idxes = torch.where(visible_num < 2)[0]
            invalid_num = invalid_idxes.size()[0]
            if invalid_num == 0:
                break

            new_positions = torch.from_numpy(self.generate_batch_positions(invalid_num)).to(self.rendering_device)
            new_n2d = torch.from_numpy(np.random.rand(invalid_num,2).astype(np.float32)).to(self.rendering_device)
            new_chosen_rt_id = np.random.rand(invalid_num,available_rt_num).argsort(axis=1)[:,:2]
            positions[invalid_idxes] = new_positions
            n_2d[invalid_idxes] = new_n2d
            chosen_rt_id[invalid_idxes.cpu().numpy()] = new_chosen_rt_id
        
        n_global = torch_render.back_full_octa_map(n_2d)#(batch,3) normal global
        theta = torch.from_numpy(np.random.rand(batch_size,1).astype(np.float32)*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]).to(self.rendering_device)
        t_global,_ = torch_render.build_frame_f_z(n_global,theta,with_theta=True)
        b_global = torch.cross(n_global,t_global)

        frame_global = [n_global,t_global,b_global]#(#(n,t,b) n is (batchsize,3))

        tmp_R_matrixs = self.R_matrixs[chosen_rt_id.reshape(-1)].reshape((batch_size,2,3,3))
        tmp_T_vecs = self.T_vecs[chosen_rt_id.reshape((-1))].reshape((batch_size,2,3,1))
        R_matrix_1 = tmp_R_matrixs[:,0]
        T_vec_1 = tmp_T_vecs[:,0]
        R_matrix_2 = tmp_R_matrixs[:,1]
        T_vec_2 = tmp_T_vecs[:,1]

        positions_1 = (torch.matmul(R_matrix_1,positions.reshape(batch_size,3,1))+T_vec_1).reshape(batch_size,3)
        positions_2 = (torch.matmul(R_matrix_2,positions.reshape(batch_size,3,1))+T_vec_2).reshape(batch_size,3)

        frame_1 = [
            torch.matmul(R_matrix_1,n_global.reshape(batch_size,3,1)).reshape(batch_size,3),
            torch.matmul(R_matrix_1,t_global.reshape(batch_size,3,1)).reshape(batch_size,3),
            torch.matmul(R_matrix_1,b_global.reshape(batch_size,3,1)).reshape(batch_size,3),
        ]

        frame_2 = [
            torch.matmul(R_matrix_2,n_global.reshape(batch_size,3,1)).reshape(batch_size,3),
            torch.matmul(R_matrix_2,t_global.reshape(batch_size,3,1)).reshape(batch_size,3),
            torch.matmul(R_matrix_2,b_global.reshape(batch_size,3,1)).reshape(batch_size,3),
        ]

        return frame_global,positions,frame_1,positions_1,frame_2,positions_2,R_matrix_1,R_matrix_2,T_vec_1,T_vec_2


    def generate_training_data(self,test_tangent_flag = False):
        tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        if tmp_params.shape[0] < self.batch_size:
            self.__preload()
            tmp_params = self.buffer_params[self.current_ptr:self.current_ptr+self.batch_size]
        self.current_ptr+=self.batch_size

        tmp_params[:,6:8] = self.__rejection_sampling_axay(test_tangent_flag)
        
        frame_global,positions_global,frame_1,positions_1,frame_2,positions_2,R_matrix_1,R_matrix_2,t_vec_1,t_vec_2 = self.generate_batch_frame(self.batch_size)
     
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

        input_params = torch.from_numpy(tmp_params).to(self.rendering_device)

        input_positions_1 = positions_1#torch.matmul(R_matrix_1_inv,torch.unsqueeze(positions_1,dim=2)-t_vec_1).reshape((-1,3))
        input_positions_2 = positions_2#torch.matmul(R_matrix_2_inv,torch.unsqueeze(positions_2,dim=2)-t_vec_2).reshape((-1,3))

        input_frame_1 = frame_1
        input_frame_2 = frame_2
        
        input_params = torch.cat((input_params,input_params),dim=0)#(2*batchsize,param_dim)
        input_positions = torch.cat((input_positions_1,input_positions_2),dim=0)#(2*batchsize,3)
        input_rt = torch.cat((rt_1,rt_2),dim=0)#(2*batchsize,12)

        input_frame = []
        for which_axis in range(3):
            tmp_axis = torch.cat((input_frame_1[which_axis],input_frame_2[which_axis]),dim=0)#(2*batchsize,1)
            input_frame.append(tmp_axis)

        input_rotate_angle = torch.zeros(2*self.batch_size,1,dtype=torch.float32,device=self.rendering_device)

        return input_params,input_positions,input_frame,input_rt,input_rotate_angle

    def generate_validating_data(self,test_tangent_flag):
        return self.generate_training_data(test_tangent_flag)