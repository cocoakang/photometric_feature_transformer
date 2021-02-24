import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DIFT_linear_projection(nn.Module):
    def __init__(self,args):
        super(DIFT_linear_projection,self).__init__()
        
        self.measurements_length = args["measurements_length"]
        self.lumitexel_length = args["lumitexel_length"]
        self.device = args["training_device"]
        self.noise_stddev = args["noise_stddev"]
        self.device_cpu = torch.device("cpu")
        self.setup_input = args["setup_input"]

        tmp_kernel = torch.empty(self.measurements_length,self.lumitexel_length)
        nn.init.xavier_uniform_(tmp_kernel)
        # tmp_kernel_np = tmp_kernel.cpu().numpy()
        # tmp_kernel_np.astype(np.float32).tofile("/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/BRDF_feature_extract/logs/sig20_init/details/params_init/0_tc.bin")
        
        # tmp_kernel_np = np.fromfile("/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/dift_extractor/runs/11_6_linear/models/0/W.bin",np.float32).reshape([8,24576,3])[:,:,0]
        # tmp_kernel = torch.from_numpy(tmp_kernel_np)

        self.kernel = torch.nn.Parameter(
            data = tmp_kernel,
            requires_grad=True
        )#(m_len,lumi_len,3)
        # self.rgb_tensor = self.setup_input.get_color_tensor(self.kernel.device).float()#(3,3,3) (light,cam,brdf)
        # self.rgb_tensor = self.rgb_tensor.permute(0,2,1).reshape(9,3)#(3,3,3) (light,brdf,cam)
    
    # def gaussian_random_matrix(self,n_components, n_features, random_state=None):
    #     components = np.random.normal(loc=0.0,
    #                                 # scale=1.0 / np.sqrt(n_components),
    #                                 scale = 0.15,
    #                                 size=(n_components, n_features))
    #     return components

    # def getW(self,lumi_len, K):
    #     random = self.gaussian_random_matrix(lumi_len, K)
    #     random_reshape = np.reshape(random, (lumi_len, K))
    #     return random_reshape

    # def init_lighting_pattern(self):
    #     W = self.getW(self.lumitexel_length, self.measurements_length)
    #     W = W.T#(measurements_length,lumitexel_length)
    #     W = torch.from_numpy(W.astype(np.float32))
    #     return W
    
    def get_lighting_patterns(self,device,withclamp=True):
        tmp_kernel = torch.nn.functional.normalize(self.kernel.to(device),dim=1)
        # tmp_kernel = torch.stack(
        #     [
        #         torch.max(tmp_kernel,torch.zeros(1,device=tmp_kernel.device)),
        #         torch.min(tmp_kernel,torch.zeros(1,device=tmp_kernel.device))
        #     ],dim=1
        # )
        # tmp_kernel = tmp_kernel.reshape(self.measurements_length,self.lumitexel_length)
        return tmp_kernel#torch.sigmoid(self.kernel.to(device))

    def compute_kernel_loss(self):
        W = self.get_lighting_patterns(self.device_cpu,withclamp=False)
        epsilon = 5e-3
        kernel_loss = torch.tanh((W-(1-epsilon))/epsilon)+torch.tanh((-W+epsilon)/epsilon)+2.0
        return torch.sum(kernel_loss)

    def forward(self,lumitexels,add_noise=True):
        '''
        lumitexels=[batch,lightnum,channel_num],channel_num should be 1
        project a batch of measurements and 
        return [batch,measurement_length,channel(1)]
        '''
        batch_size = lumitexels.size()[0]
        device = lumitexels.device

        measurement_list = []
        
        kernel = self.get_lighting_patterns(device)#[measurementnum,lightnum]
        tmp_kernel = torch.unsqueeze(kernel,dim=0).repeat(batch_size,1,1)#[batchsize,measurementnum,lightnum]
        tmp_measurement = torch.matmul(tmp_kernel,lumitexels)#[batch,measurement,1]

        if add_noise:
            tmp_noise = torch.randn_like(tmp_measurement)*self.noise_stddev+1.
            tmp_measurements_noised = tmp_measurement*tmp_noise
            return tmp_measurements_noised
        else:
            return tmp_measurement