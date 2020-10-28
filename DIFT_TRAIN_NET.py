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
from ALBEDO_NET import ALBEDO_NET
from DIFT_NET import DIFT_NET

class DIFT_TRAIN_NET(nn.Module):
    def __init__(self,args):
        super(DIFT_TRAIN_NET,self).__init__()

        ########################################
        ##parse configuration                ###
        ########################################
        self.training_device = args["training_device"]
        # self.sample_view_num = args["sample_view_num"]
        self.measurements_length = args["measurements_length"]
        self.batch_size = args["batch_size"]
        self.dift_code_len = args["dift_code_len"]

        self.lambdas = args["lambdas"]
        self.partition = args["partition"]
        self.dift_code_config = args["dift_code_config"]
        self.training_mode = args["training_mode"]
        ########################################
        ##loading setup configuration        ###
        ########################################
        self.setup = args["setup_input"]
        
        
        ########################################
        ##define net modules                 ###
        ########################################

        # self.denoising_net = SIGA20_NET_denoising(args)
        self.linear_projection = DIFT_linear_projection(args)
        # self.albedo_net = ALBEDO_NET(args)
        self.dift_net = DIFT_NET(args)
        # self.material_net = SIGA20_NET_material(args)
        # self.decompose_net = SIGA20_NET_m_decompose(args)
        self.l2_loss_fn = torch.nn.MSELoss(reduction='sum')
        # self.l2_loss_fn_none = torch.nn.MSELoss(reduction='none')
        # self.regularizer = Regularization(self.dift_net,1.0)#self.reg_alpha)

        self.diag_ind = np.diag_indices(self.dift_code_len)

        # self.bn = torch.nn.BatchNorm1d(self.dift_code_len, affine=False)


    def compute_e1_loss(self,Y1,Y2):
        #E1
        eps = 1e-6
        Y1_tmp = torch.unsqueeze(Y1,dim=0)
        Y2_tmp = torch.unsqueeze(Y2,dim=1)
        D_sub = Y1_tmp-Y2_tmp#(batch,batch,diftcode_len)
        D = torch.sqrt(torch.sum(D_sub*D_sub,dim=2)+1e-6)
        # D_mat_mul = torch.matmul(Y1,Y2.T)#(batch,batch)
        # D = torch.sqrt(2.0*(1.0-D_mat_mul+1e-6))#[batch,batch]               

        D_exp = torch.exp(-D)
        D_ii = torch.unsqueeze(torch.diag(D_exp),dim=1)#[batch,1]
        #"compute_col_loss"
        D_col_sum = torch.sum(D_exp.T,dim=1,keepdim=True)#[batch,1]
        s_ii_c = D_ii / (eps+D_col_sum)
        #"compute_row_loss"
        D_row_sum = torch.sum(D_exp,dim=1,keepdim=True)#[batch,1]
        s_ii_r = D_ii / (eps+D_row_sum)
        
        tmp_E1 = -0.5*(torch.sum(torch.log(s_ii_c))+torch.sum(torch.log(s_ii_r)))
    
        return tmp_E1,D

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
        global_positions = batch_data["position"].to(self.training_device)#(batchsize,3)
        normal_label = batch_data["normal"].to(self.training_device)#(2*batchsize,3)
        rotate_theta = batch_data["rotate_theta"].to(self.training_device)#(2*batchsize,1)
        
        ############################################################################################################################
        ## step 2 draw nn net
        ############################################################################################################################
        #1 first we project every lumitexel to measurements
        input_lumis = input_lumis.reshape(2*self.batch_size,self.setup.get_light_num(),1)
        measurements = self.linear_projection(input_lumis)#(2*batchsize,m_len,1)
        # measurements_for_albedo = measurements[:,:self.partition["albedo"][0]]
        # measurements_for_dift = measurements[:,self.partition["albedo"][0]:]
        #concatenate measurements
        
        #2 infer albedo using neural network
        # albedo_nn_diff,albedo_nn_spec = self.albedo_net(measurements_for_albedo)#(2*batchsize,1),(2*batchsize,1)

        # view_mat_model = torch_render.rotation_axis(-rotate_theta,self.setup.get_rot_axis_torch(measurements.device))#[2*batch,4,4]
        # view_mat_model_t = torch.transpose(view_mat_model,1,2)#[2*batch,4,4]
        # view_mat_model_t = view_mat_model_t.reshape(2*self.batch_size,16)
        # view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
        # view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[2*batch,4,4]
        # view_mat_for_normal_t = view_mat_for_normal_t.reshape(2*self.batch_size,16)

        cossin = torch.cat(
            [
                torch.sin(rotate_theta),
                torch.cos(rotate_theta)
            ],dim=1
        )

        # dift_codes_full,origin_codes_map = self.dift_net(measurements,view_mat_model_t,view_mat_for_normal_t,param_2[:,[5]],param_2[:,[6]],True)#(2*batch,diftcodelen)
        dift_codes_full,origin_codes_map = self.dift_net(measurements,cossin,True)#(2*batch,diftcodelen)
        dift_codes_full = dift_codes_full.reshape(2,self.batch_size,self.dift_code_len)
        ############################################################################################################################
        ## step 3 compute loss
        ############################################################################################################################
        E1_loss_map = {}
        if call_type == "train":
            if self.training_mode == "pretrain":
                for i,code_key in enumerate(origin_codes_map):
                    dift_codes = origin_codes_map[code_key].reshape(2,self.batch_size,self.dift_code_config[code_key][0])
                    
                    Y1 = dift_codes[0]#[batch,diftcode_len]
                    Y2 = dift_codes[1]#[batch,diftcode_len]
                
                    tmp_E1,_ = self.compute_e1_loss(Y1,Y2)
                
                    E1_loss_map[code_key+"_loss"] = tmp_E1.item()
                    if i == 0:
                        E1 = tmp_E1*self.dift_code_config[code_key][1]
                    else:
                        E1 = E1 + tmp_E1*self.dift_code_config[code_key][1]
            elif self.training_mode == "finetune":
                Y1 = dift_codes_full[0]#[batch,diftcode_len]
                Y2 = dift_codes_full[1]#[batch,diftcode_len]

                E1,_ = self.compute_e1_loss(Y1,Y2)

        elif call_type == "val" or call_type == "check_quality":
            Y1 = dift_codes_full[0]#[batch,diftcode_len]
            Y2 = dift_codes_full[1]#[batch,diftcode_len]
            
            E1,D = self.compute_e1_loss(Y1,Y2)

            if call_type == "check_quality" :
                term_map = {
                    "input_lumis":input_lumis.cpu(),
                    "distance_matrix":D.cpu(),
                    "lighting_pattern":self.linear_projection.get_lighting_patterns(self.training_device),
                    "global_positions":global_positions.cpu(),
                    "normal_label":normal_label.cpu(),
                    "normal_nn":normal_label.cpu()
                }
                term_map = self.visualize_quality_terms(term_map)
                return term_map
        else:
            print("unkown call type")
            exit(0)
        #######
        #### covariance loss
        #######
        # print("========================================")
        # Y1 = dift_codes_full[0]#[batch,diftcode_len]
        # Y2 = dift_codes_full[1]#[batch,diftcode_len]
        # for i,Ys in enumerate([Y1, Y2]):
        #     '''
        #     Ys (imgnum,codelen)
        #     '''
        #     Ys = self.bn(Ys)
        #     Rs = torch.matmul(Ys.permute(1,0),Ys)/self.dift_code_len
        #     Rs[self.diag_ind[0],self.diag_ind[1]] = 0.0
        #     tmp_E2 = torch.sum(Rs*Rs)*0.5

        #     if i == 0:
        #         E2 = tmp_E2
        #     else:
        #         E2 = E2 + tmp_E2

        #######
        #### albedo loss
        #######
        # albedo_loss_diff = self.l2_loss_fn(albedo_nn_diff,param_2[:,[5]])
        # albedo_loss_spec = self.l2_loss_fn(albedo_nn_spec,param_2[:,[6]])
        # albedo_loss = albedo_loss_diff*self.lambdas["albedo_diff"]+albedo_loss_spec*self.lambdas["albedo_spec"]

        # if global_step > 1:
        #     exit(0)
        ###material loss
        # D_exp_m = D_exp[self.batch_brdf_num*0:self.batch_brdf_num*1,self.batch_size-self.batch_brdf_num*1:self.batch_size]
        # l2_loss_m = torch.sum(torch.diag(D_exp_m))*0.05

        ### !6 reg loss
        # reg_loss = self.regularizer(self.dift_net)
        total_loss = E1*self.lambdas["E1"]#+E2*self.lambdas["E2"]

        loss_log_map = {
            # "albedo_value_total_loss":albedo_loss.item(),
            # "albedo_value_diff_loss":albedo_loss_diff.item(),
            # "albedo_value_spec_loss":albedo_loss_spec.item(),
            "e1_loss":E1.item(),
            # "e2_loss":E2.item(),
            "total_loss":total_loss.item(),
        }
        loss_log_map.update(E1_loss_map)
        return total_loss,loss_log_map
    
    def visualize_quality_terms(self,quality_map):
        result_map = {}
        #################################
        ###training samples
        #################################
        input_lumis = quality_map["input_lumis"].numpy()#(2*batchsize,lightnum,channel) torch tensor
        distance_matrix = quality_map["distance_matrix"].numpy()
        global_positions = quality_map["global_positions"].numpy()#(batchsize,3)
        normal_label = quality_map["normal_label"].numpy()#(2*batchsize,3)
        normal_nn = quality_map["normal_nn"].numpy()#(2*batchsize,3)
        
        img_stack_list = []
        input_lumis = np.transpose(input_lumis.reshape(2,self.batch_size,input_lumis.shape[1],input_lumis.shape[2]),[1,0,2,3])
        input_lumis = np.reshape(input_lumis,[self.batch_size*2,input_lumis.shape[2],input_lumis.shape[3]])
        img_input = torch_render.visualize_lumi(input_lumis,self.setup)#(batchsize*2,imgheight,imgwidth,channel)

        normal_label = np.reshape(np.transpose(np.reshape(normal_label,(2,self.batch_size,3)),[1,0,2]),(self.batch_size*2,3))
        global_positions = np.reshape(np.repeat(np.expand_dims(global_positions,axis=1),2,axis=1),(self.batch_size*2,3))
        img_input = torch_render.draw_vector_on_lumi(img_input,normal_label,global_positions,self.setup,True,(0.0,1.0,0.0))
        normal_nn = np.transpose(normal_nn.reshape(2,self.batch_size,3),[1,0,2])
        normal_nn = np.reshape(normal_nn,(self.batch_size*2,3))
        img_input = torch_render.draw_vector_on_lumi(img_input,normal_nn,global_positions,self.setup,True,(1.0,0.0,0.0))

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
        row_max = np.repeat(np.max(distance_matrix,axis=1,keepdims=True),self.batch_size,axis=1)
        col_max = np.repeat(np.max(distance_matrix,axis=0,keepdims=True),self.batch_size,axis=0)
        row_col_max = np.max(np.stack((row_max,col_max),axis=-1),axis=-1)+1e-6
        distance_matrix = distance_matrix / row_col_max
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