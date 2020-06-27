import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils
import torchvision
import time
import numpy as np
import math
import sys
from weight_regularizer import Regularization

TORCH_RENDER_PATH="../torch_renderer/"

sys.path.append(TORCH_RENDER_PATH)
import torch_render
from multiview_renderer_mp import Multiview_Renderer
from torch_render import Setup_Config
from DIFT_linear_projection import DIFT_linear_projection
from DIFT_NET import DIFT_NET

class DIFT_TRAIN_NET(nn.Module):
    def __init__(self,args):
        super(DIFT_TRAIN_NET,self).__init__()

        ########################################
        ##parse configuration                ###
        ########################################
        self.training_device = args["training_device"]
        self.sample_view_num = args["sample_view_num"]
        self.measurements_length = args["measurements_length"]
        self.batch_size = args["batch_size"]
        self.dift_code_len = args["dift_code_len"]

        self.lambdas = args["lambdas"]

        ########################################
        ##loading setup configuration        ###
        ########################################
        self.setup = args["setup_input"]
        
        
        ########################################
        ##define net modules                 ###
        ########################################

        # self.denoising_net = SIGA20_NET_denoising(args)
        self.linear_projection = DIFT_linear_projection(args)
        self.dift_net = DIFT_NET(args)
        # self.material_net = SIGA20_NET_material(args)
        # self.decompose_net = SIGA20_NET_m_decompose(args)
        self.l2_loss_fn = torch.nn.MSELoss(reduction='sum')
        # self.l2_loss_fn_none = torch.nn.MSELoss(reduction='none')
        # self.regularizer = Regularization(self.material_net,self.lambdas["weight"])#self.reg_alpha)


    def forward(self,batch_data,call_type="train",global_step=-1):
        '''
        batch_data = map of data [batch,-1]
        '''
        # input_positions = torch.from_numpy(batch_data["input_positions"]).to(self.training_device)
        # multiview_lumitexel_list = [torch.from_numpy(a).to(self.training_device) for a in batch_data["multiview_lumitexel_list"]]
        # rendered_slice_diff_gt_list = [torch.from_numpy(a).to(self.training_device) for a in batch_data["rendered_slice_diff_gt_list"]]
        # rendered_slice_spec_gt_list = [torch.from_numpy(a).to(self.training_device) for a in batch_data["rendered_slice_spec_gt_list"]]
        # normal_label = torch.from_numpy(batch_data["normal_label"]).to(self.training_device)
        # geometry_normal = torch.from_numpy(batch_data["geometry_normal"]).to(self.training_device)
        # start = time.time()
        # end_time = [start]
        # stamp_name=[]
        input_lumis = batch_data["input_lumi"].to(self.training_device)#(2,batchsize,lumi_len,3)

        ############################################################################################################################
        ## step 2 draw nn net
        ############################################################################################################################
        #1 first we project every lumitexel to measurements
        input_lumis = input_lumis.reshape(2*self.batch_size,self.setup.get_light_num(),3)
        measurements = self.linear_projection(input_lumis)#(2*batchsize,m_len,3)
        
        #concatenate measurements
        dift_codes = self.dift_net(measurements)#(2*batch,diftcodelen)
        dift_codes = dift_codes.reshape(2,self.batch_size,self.dift_code_len)

        ############################################################################################################################
        ## step 3 compute loss
        ############################################################################################################################
        Y1 = dift_codes[0]#[batch,diftcode_len]
        Y2 = dift_codes[1]#[batch,diftcode_len]
        
        #E1
        eps = 1e-6
        D_mat_mul = torch.matmul(Y1,Y2.T)#(batch,batch)
        D = torch.sqrt(2.0*(1.0-D_mat_mul+1e-6))#[batch,batch]
        
        if call_type == "check_quality":
            #fech every lumitexel from gpus
            term_map = {
                "input_lumis":input_lumis.cpu(),
                "distance_matrix":D.cpu(),
                "lighting_pattern":self.linear_projection.get_lighting_patterns(self.training_device)
            }
            term_map = self.visualize_quality_terms(term_map)
            return term_map


        D_exp = torch.exp(2.0-D)
        D_ii = torch.unsqueeze(torch.diag(D_exp),dim=1)#[batch,1]
        #"compute_col_loss"
        D_col_sum = torch.sum(D_exp.T,dim=1,keepdim=True)#[batch,1]
        s_ii_c = D_ii / (eps+D_col_sum)
        #"compute_row_loss"
        D_row_sum = torch.sum(D_exp,axis=1,keepdim=True)#[batch,1]
        s_ii_r = D_ii / (eps+D_row_sum)
        
        E1 = -0.5*(torch.sum(torch.log(s_ii_c))+torch.sum(torch.log(s_ii_r)))

        # if torch.isnan(E1).any() or global_step == 14525:
        #     print("=====================nan occured!")
        #     print("global step:",global_step)
        #     print("D_mat_mul\n",D_mat_mul)
        #     print("D\n",D)
        #     print("D_ii\n",D_ii)
        #     print("s_ii_c\n",s_ii_c)
        #     print("s_ii_r\n",s_ii_r)
        #     # exit()

        l2_loss =   E1

        ### !6 reg loss
        # reg_loss = self.regularizer(self.material_net)

        total_loss = l2_loss#+reg_loss#+loss_kernel.to(l2_loss.device)*0.03

        loss_log_map = {
            "loss_e1_train_tamer":E1.item(),
            "total":total_loss.item()
        }
        return total_loss,loss_log_map
    
    def visualize_quality_terms(self,quality_map):
        result_map = {}
        #################################
        ###training samples
        #################################
        input_lumis = quality_map["input_lumis"].numpy()#(2*batchsize,lightnum,channel) torch tensor
        distance_matrix = quality_map["distance_matrix"].numpy()
        
        img_stack_list = []
        input_lumis = np.transpose(input_lumis.reshape(2,self.batch_size,input_lumis.shape[1],input_lumis.shape[2]),[1,0,2,3])
        input_lumis = np.reshape(input_lumis,[self.batch_size*2,input_lumis.shape[2],input_lumis.shape[3]])
        img_input = torch_render.visualize_lumi(input_lumis,self.setup)#(batchsize*2,imgheight,imgwidth,channel)
        img_input = torch.from_numpy(img_input).reshape(self.batch_size,2,img_input.shape[1],img_input.shape[2],img_input.shape[3])

        images_list = []
        for which_sample in range(self.batch_size):
            tmp_lumi_img = torchvision.utils.make_grid(img_input[which_sample].permute(0,3,1,2),nrow=2, pad_value=0.5)
            
            images_list.append(tmp_lumi_img)

        images = torch.stack(images_list,axis=0)#(batchsize,3,height(with padding),width(widthpadd))
        images = torch.clamp(images,0.0,1.0)

        result_map["multiview_lumi_img"] = images

        #################################
        ###distance matrix
        #################################
        distance_matrix = distance_matrix / np.max(distance_matrix)
        distance_matrix = np.repeat(np.expand_dims(distance_matrix,axis=2),3,axis=2)
        distance_matrix = np.transpose(distance_matrix,[2,0,1])
        result_map["distance_matrix"] = torch.from_numpy(distance_matrix)
        
        #################################
        ###lighting patterns
        #################################
        lighting_pattern = [a_kernel.cpu().detach().numpy() for a_kernel in quality_map["lighting_pattern"]]
        
        lighting_pattern_collector = []
        for which_m in range(len(lighting_pattern)):
            lp_pos = np.maximum(0.0,lighting_pattern[which_m])
            lp_pos_max = (lp_pos.max()+1e-6)
            lp_pos = torch_render.visualize_lumi(lp_pos/lp_pos_max,self.setup,is_batch_lumi=False)#(imgheight,imgwidth,3)
            lp_neg = -np.minimum(0.0,lighting_pattern[which_m])
            lp_neg_max = (lp_neg.max()+1e-6)
            lp_neg = torch_render.visualize_lumi(lp_neg/lp_neg_max,self.setup,is_batch_lumi=False)#(imgheight,imgwidth,3)

            lighting_pattern_collector.append(lp_pos)
            lighting_pattern_collector.append(lp_neg)


        lighting_pattern_collector = np.array(lighting_pattern_collector).reshape([self.measurements_length,2,*(self.setup.img_size),3])#(lightingpattern_num,2,img_height,img_width,3)
        result_map["lighting_pattern_rgb"] = lighting_pattern_collector

        return result_map