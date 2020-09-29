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
from DIFT_NET_v import DIFT_NET_V
from DIFT_NET_h import DIFT_NET_H
from DIFT_NET_m import DIFT_NET_M

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
        self.batch_brdf_num = args["batch_brdf_num"]
        self.dift_code_len = args["dift_code_len"]
        self.dift_code_len_gv = args["dift_code_len_gv"]
        self.dift_code_len_gh = args["dift_code_len_gh"]
        self.dift_code_len_m = args["dift_code_len_m"]

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
        self.dift_net_gh = DIFT_NET_H(args)
        self.dift_net_gv = DIFT_NET_V(args)
        self.dift_net_m = DIFT_NET_M(args)
        # self.material_net = SIGA20_NET_material(args)
        # self.decompose_net = SIGA20_NET_m_decompose(args)
        self.l2_loss_fn = torch.nn.MSELoss(reduction='sum')
        # self.l2_loss_fn_none = torch.nn.MSELoss(reduction='none')
        # self.regularizer = Regularization(self.dift_net,1.0)#self.reg_alpha)

        self.diag_ind = np.diag_indices(self.dift_code_len)

        self.bn = torch.nn.BatchNorm1d(self.dift_code_len, affine=False)


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
        view_ids_cossin = batch_data["view_ids_cossin"].to(self.training_device)#(2*batchsize,2)
        global_positions = batch_data["position"].to(self.training_device)#(batchsize,3)
        normal_label = batch_data["normal"].to(self.training_device)#(2*batchsize,3)
        normal_label_local = batch_data["normal_local"].to(self.training_device)#(2*batchsize,3)
        rotate_theta = batch_data["rotate_theta"].to(self.training_device)#(2*batchsize,1)
        position_2 = batch_data["position_2"].to(self.training_device)#(2*batchsize,3)

        ############################################################################################################################
        ## step 2 draw nn net
        ############################################################################################################################
        #1 first we project every lumitexel to measurements
        input_lumis = input_lumis.reshape(2*self.batch_size,self.setup.get_light_num(),1)
        measurements = self.linear_projection(input_lumis)#(2*batchsize,m_len,1)
        #concatenate measurements

        view_mat_model = torch_render.rotation_axis(-rotate_theta,self.setup.get_rot_axis_torch(measurements.device))#[2*batch,4,4]
        view_mat_model_t = torch.transpose(view_mat_model,1,2)#[2*batch,4,4]
        view_mat_model_t = view_mat_model_t.reshape(2*self.batch_size,16)
        view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
        view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[2*batch,4,4]
        view_mat_for_normal_t = view_mat_for_normal_t.reshape(2*self.batch_size,16)

        dift_codes_gh_origin = self.dift_net_gh(measurements[:,0:5,:],view_ids_cossin,view_mat_model_t,view_mat_for_normal_t)#(2*batch,diftcodelen)
        dift_codes_gv_origin = self.dift_net_gv(measurements[:,5:10,:],view_ids_cossin,view_mat_model_t,view_mat_for_normal_t)#(2*batch,diftcodelen)
        dift_codes_m_origin = self.dift_net_m(measurements[:,10:,:],view_ids_cossin,view_mat_model_t,view_mat_for_normal_t)#(2*batch,diftcodelen)
        # dift_codes_origin = dift_codes_origin*0.0+position_2
        # dift_codes_origin = torch_render.rotate_point_along_axis(self.setup,-rotate_theta,dift_codes_origin)
        dift_codes_gh_origin = dift_codes_gh_origin.reshape(2,self.batch_size,self.dift_code_len_gh)
        dift_codes_gv_origin = dift_codes_gv_origin.reshape(2,self.batch_size,self.dift_code_len_gv)
        dift_codes_m_origin = dift_codes_m_origin.reshape(2,self.batch_size,self.dift_code_len_m)

        dift_codes_full = torch.cat([dift_codes_gh_origin,dift_codes_gv_origin,dift_codes_m_origin],dim=2)
        ############################################################################################################################
        ## step 3 compute loss
        ############################################################################################################################
        if call_type == "train":
            E1_collector = []
            for dift_codes in [dift_codes_gh_origin,dift_codes_gv_origin,dift_codes_m_origin]:
                Y1 = dift_codes[0]#[batch,diftcode_len]
                Y2 = dift_codes[1]#[batch,diftcode_len]
                
                #E1
                eps = 1e-6
                Y1_tmp = torch.unsqueeze(Y1,dim=0)
                Y2_tmp = torch.unsqueeze(Y2,dim=1)
                D_sub = Y1_tmp-Y2_tmp#(batch,batch,diftcode_len)
                D = torch.sqrt(torch.sum(D_sub*D_sub,dim=2)+1e-6)
                # D_mat_mul = torch.matmul(Y1,Y2.T)#(batch,batch)
                # D = torch.sqrt(2.0*(1.0-D_mat_mul+1e-6))#[batch,batch]               

                D_exp = torch.exp(2.0-D)
                D_ii = torch.unsqueeze(torch.diag(D_exp),dim=1)#[batch,1]
                #"compute_col_loss"
                D_col_sum = torch.sum(D_exp.T,dim=1,keepdim=True)#[batch,1]
                s_ii_c = D_ii / (eps+D_col_sum)
                #"compute_row_loss"
                D_row_sum = torch.sum(D_exp,dim=1,keepdim=True)#[batch,1]
                s_ii_r = D_ii / (eps+D_row_sum)
                
                tmp_E1 = -0.5*(torch.sum(torch.log(s_ii_c))+torch.sum(torch.log(s_ii_r)))
                E1_collector.append(tmp_E1)

            E1 = E1_collector[0]+E1_collector[1]*1e1+E1_collector[2]*1e-1
        elif call_type == "check_quality":
            Y1 = dift_codes_full[0]#[batch,diftcode_len]
            Y2 = dift_codes_full[1]#[batch,diftcode_len]
            
            #E1
            eps = 1e-6
            Y1_tmp = torch.unsqueeze(Y1,dim=0)
            Y2_tmp = torch.unsqueeze(Y2,dim=1)
            D_sub = Y1_tmp-Y2_tmp#(batch,batch,diftcode_len)
            D = torch.sqrt(torch.sum(D_sub*D_sub,dim=2)+1e-6)
            # D_mat_mul = torch.matmul(Y1,Y2.T)#(batch,batch)
            # D = torch.sqrt(2.0*(1.0-D_mat_mul+1e-6))#[batch,batch]        

            #fech every lumitexel from gpus
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
        elif call_type == "val":
            Y1 = dift_codes_full[0]#[batch,diftcode_len]
            Y2 = dift_codes_full[1]#[batch,diftcode_len]
            
            #E1
            eps = 1e-6
            Y1_tmp = torch.unsqueeze(Y1,dim=0)
            Y2_tmp = torch.unsqueeze(Y2,dim=1)
            D_sub = Y1_tmp-Y2_tmp#(batch,batch,diftcode_len)
            D = torch.sqrt(torch.sum(D_sub*D_sub,dim=2)+1e-6)

            D_exp = torch.exp(2.0-D)
            D_ii = torch.unsqueeze(torch.diag(D_exp),dim=1)#[batch,1]
            #"compute_col_loss"
            D_col_sum = torch.sum(D_exp.T,dim=1,keepdim=True)#[batch,1]
            s_ii_c = D_ii / (eps+D_col_sum)
            #"compute_row_loss"
            D_row_sum = torch.sum(D_exp,dim=1,keepdim=True)#[batch,1]
            s_ii_r = D_ii / (eps+D_row_sum)
            
            E1 = -0.5*(torch.sum(torch.log(s_ii_c))+torch.sum(torch.log(s_ii_r)))
        else:
            print("unkown call type")
            exit(0)
        # covariance loss
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

        l2_loss = E1#+E2#position_loss
        # if global_step > 1:
        #     exit(0)
        ###material loss
        # D_exp_m = D_exp[self.batch_brdf_num*0:self.batch_brdf_num*1,self.batch_size-self.batch_brdf_num*1:self.batch_size]
        # l2_loss_m = torch.sum(torch.diag(D_exp_m))*0.05

        ### !6 reg loss
        # reg_loss = self.regularizer(self.dift_net)

        total_loss = l2_loss#+reg_loss#+loss_kernel.to(l2_loss.device)*0.03

        loss_log_map = {
            "loss_e1_train_tamer":E1.item(),
            # "loss_e2_train_tamer":E2.item(),
            # "loss_normal":0.0,
            "total":total_loss.item(),
            # "loss_reg_tamer":reg_loss.item()
        }
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