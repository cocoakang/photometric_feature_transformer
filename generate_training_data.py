import numpy as np
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter 
import torchvision
import sys
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from torch_render import Setup_Config
from multiview_renderer_mt import Multiview_Renderer
import os
import time
import argparse
import mine

def compute_loss_weight(batch_data,sample_view_num):
    '''
    input is cos(theta)
    '''
    #k=(cos(pi/sample_view_num))
    #0.5*sin(pi/k*batch_data-pi/2)+0.5
    k = math.cos(math.pi*0.5-2.0*math.pi/sample_view_num)
    origin = 0.5*torch.sin(math.pi/k*batch_data-math.pi/2.0)+0.5
    origin = torch.where(batch_data>k,torch.ones_like(batch_data),origin)
    origin = torch.where(batch_data<0.0,torch.zeros_like(batch_data),origin)
    return origin

def compute_loss_weight_new(normal,wo,decay_range,concentrate_ratio=1.0):
    '''
    normal is (batchsize,3) unit vectors
    wo is (batchsize,3) unit vectors
    decay_theta is in radian
    '''
    d = torch.abs(normal[:,[2]]*wo[:,[2]])/torch.sqrt(torch.clamp(normal[:,[0]]*normal[:,[0]]+normal[:,[1]]*normal[:,[1]],min=1e-8))
    # d = (normal[:,[2]]*wo[:,[2]])/(normal[:,[1]]*normal[:,[1]])/torch.sqrt(((normal[:,[0]]*normal[:,[0]])/(normal[:,[1]]*normal[:,[1]])+1.0))
    r = torch.sqrt(torch.clamp(1-wo[:,[2]]*wo[:,[2]],min=1e-8))#(batch,1)
    # print("d:",d)
    # print("r",r)
    d = torch.where(d>r,r,d)
    
    m = 2.0*torch.sqrt(r*r-d*d)
    # print("m:",m)
    alpha = torch.acos((2*r*r-m**2)/(2*r*r))#TODO clamp to -1.0,1.0
    
    # print(alpha)
    alpha = torch.where(normal[:,[2]] > 0.0,alpha,2*math.pi-alpha)#invisible angle
    alpha = torch.where(wo[:,[2]] < 0.0,2.0*math.pi - alpha,alpha)
    alpha = alpha*0.5
    end_theta_origin = math.pi-alpha#visible angle
    # test = end_theta
    # print("end_theta:",end_theta.reshape(-1,50)[:30,-1]/math.pi*180.0)

    # max1_start = 0.0
    max1_end_origin = end_theta_origin-decay_range
    max1_end = torch.clamp(max1_end_origin,min=0.0)
    # print("max1_end:",max1_end/math.pi*180.0)

    B = torch.cos(math.pi*0.5-torch.acos(torch.abs(wo[:,[2]])))#nz_bound
    # print("B:",torch.acos(B)/math.pi*180.0)
    # print("normal:",torch.acos(torch.abs(normal[:,[2]]))/math.pi*180.0)
    shift = decay_range/(1-B)*(torch.abs(normal[:,[2]])-B)     
    # print("shift:",shift/math.pi*180.0)
    shift = torch.where(torch.abs(normal[:,[2]]) == 1.0,torch.ones_like(shift).fill_(decay_range),shift)
    shift = torch.where(end_theta_origin < (math.pi-1e-4),torch.zeros_like(shift),shift)# 
    # print("shift:",shift.reshape(-1,50)[:14,-1]/math.pi*180.0)

    max1_end = max1_end+shift#max1 end without any adjustment
    max1_end_origin = max1_end_origin+shift
    
    # in_north = math.pi-max1_end > decay_range
    
    #adjust concentrate here!!!
    max1_end = max1_end*concentrate_ratio
    max1_end_origin = max1_end_origin*concentrate_ratio
    end_theta = max1_end_origin+decay_range
    end_theta = torch.where(end_theta_origin <1e-6,torch.zeros_like(end_theta),end_theta    )

    # print("beta:",beta)
    normal_xy = normal.clone()
    normal_xy[:,2] = 0.0
    normal_xy = torch.nn.functional.normalize(normal_xy)
    wo_xy = wo.clone()
    wo_xy[:,2] = 0.0
    wo_xy = torch.nn.functional.normalize(wo_xy)
    theta = torch.acos(torch.clamp(torch.sum(normal_xy*wo_xy,dim=1,keepdims=True),min=-1.0,max=1.0))

    # loss_weight = (end_theta-theta+shift)/decay_range
    loss_weight = 0.5*torch.cos(math.pi/(end_theta-max1_end_origin)*theta+math.pi*max1_end_origin/(max1_end_origin-end_theta))+0.5#(end_theta-theta+shift)/decay_range

    loss_weight = torch.where(theta <= max1_end,torch.ones_like(loss_weight),loss_weight)
    loss_weight = torch.where(theta > end_theta ,torch.zeros_like(loss_weight),loss_weight)
    
    # loss_weight = torch.where(loss_weight > 0.0 ,torch.ones_like(loss_weight),torch.zeros_like(loss_weight))
    # loss_weight = torch.where(
    #     end_theta >= math.pi-1e-4,
    #     torch.clamp(loss_weight+decay_range/(1-B)*shift,max=1.0),
    #     loss_weight
    # )

    # print("loss_weight:",loss_weight)
    return loss_weight

def generate_batch_data(batch_size,device,args):
    #origin data:
    n_2d = np.random.uniform(mine.param_bounds["n"][0],mine.param_bounds["n"][1],[batch_size,2]).astype(np.float32)
    theta = np.random.uniform(mine.param_bounds["theta"][0],mine.param_bounds["theta"][1],[batch_size,1]).astype(np.float32)
    axay = mine.rejection_sampling_axay(batch_size).astype(np.float32)
    pd = np.random.uniform(mine.param_bounds["pd"][0],mine.param_bounds["pd"][1],[batch_size,1]).astype(np.float32)
    ps = np.random.uniform(mine.param_bounds["ps"][0],mine.param_bounds["ps"][1],[batch_size,1]).astype(np.float32)
    position = np.concatenate([
        np.random.uniform(mine.param_bounds["box"][0],mine.param_bounds["box"][1],[batch_size,2]),
        np.random.uniform(-30.0,120.0,[batch_size,1])
    ],axis=-1).astype(np.float32)

    n_2d = torch.from_numpy(n_2d).to(device)
    theta = torch.from_numpy(theta).to(device)
    axay = torch.from_numpy(axay).to(device)
    pd = torch.from_numpy(pd).to(device)
    ps = torch.from_numpy(ps).to(device)
    input_positions = torch.from_numpy(position).to(device)
    input_params = torch.cat([n_2d,theta,axay,pd,ps],dim=1)

    #compute noised data here
    sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=args.sample_view_num,endpoint=False)#[self.sample_view_num]
    sampled_rotate_angles_np = (np.expand_dims(sampled_rotate_angles_np,axis=0)).astype(np.float32)
    sampled_rotate_angles_np = np.repeat(sampled_rotate_angles_np,batch_size,axis=0)#[batch,sample_view_num]
    sampled_view_angles = torch.from_numpy(sampled_rotate_angles_np).to(device)

    while True:
        normal = torch_render.back_full_octa_map(n_2d)
        v_c = []
        for which_view in range(args.sample_view_num):    
            v_c.append(torch_render.compute_wo_dot_n(args.setup,input_positions,sampled_view_angles[:,[which_view]],normal,args.setup.get_cam_pos_torch(device)))
        v_c = torch.cat(v_c,dim=1)#(batch,sample_view_num)
        still_needed = torch.where(torch.sum(v_c>0.0,dim=1) == 0)[0]
        if len(still_needed) == 0:
            break
        n_2d[still_needed] = torch.from_numpy(np.random.uniform(mine.param_bounds["n"][0],mine.param_bounds["n"][1],[len(still_needed),2]).astype(np.float32)).to(device)
    tangent,binormal = torch_render.build_frame_f_z(normal,theta,with_theta=True)

    global_frame = [normal,tangent,binormal]

    #####build geometry normal
    tmp_x2 = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(device)
    disturb_len = torch.from_numpy(np.random.normal(0.0,args.disturb_stddev["geo_normal"],[batch_size,1]).astype(np.float32)).to(device)
    
    while True:
        disturb_dir = global_frame[1]*tmp_x2[:,[0]]+global_frame[2]*tmp_x2[:,[1]]
        normal_geo = global_frame[0]+disturb_dir*disturb_len
        normal_geo = torch.nn.functional.normalize(normal_geo)
        v_c = []
        for which_view in range(args.sample_view_num):    
            v_c.append(torch_render.compute_wo_dot_n(args.setup,input_positions,sampled_view_angles[:,[which_view]],normal_geo,args.setup.get_cam_pos_torch(device)))
        v_c = torch.cat(v_c,dim=1)#(batch,sample_view_num)
        still_needed = torch.where(torch.sum(v_c>0.0,dim=1) == 0)[0]
        if len(still_needed) == 0:
            break
        tmp_x2[still_needed] = torch.from_numpy(np.random.rand(len(still_needed),2).astype(np.float32)).to(device)
        disturb_len[still_needed] = torch.from_numpy(np.random.normal(0.0,args.disturb_stddev["geo_normal"],[len(still_needed),1]).astype(np.float32)).to(device)

    global_frame_noised_list = []
    input_params_noised_list = []
    input_positions_noised_list = []
    is_visible_origin_flag_list = []
    
    for which_view in range(args.sample_view_num):
        origin_frame = global_frame
        origin_param = input_params
        origin_position = input_positions

        #COMPUTE VISIBILITY USING GEOMETRY NORMAL
        wo_dot_n = torch_render.compute_wo_dot_n(args.setup,origin_position,sampled_view_angles[:,[which_view]],origin_frame[0],args.setup.get_cam_pos_torch(device))#(batch,1)
        visible_flag_origin = wo_dot_n > 0.0

        wo_dot_n_geo = torch_render.compute_wo_dot_n(args.setup,origin_position,sampled_view_angles[:,[which_view]],normal_geo,args.setup.get_cam_pos_torch(device))#(batch,1)
        visible_flag_geo = wo_dot_n_geo > 0.0

        n_local = torch_render.back_hemi_octa_map(torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(device))
        theta_hogwild = torch.from_numpy(np.random.uniform(mine.param_bounds["theta"][0],mine.param_bounds["theta"][1],[batch_size,1]).astype(np.float32)).to(device)
        t_local,b_local = torch_render.build_frame_f_z(n_local,theta_hogwild)
        n_frame = torch.unsqueeze(args.setup.get_cam_pos_torch(device),dim=0)-torch_render.rotate_point_along_axis(args.setup,sampled_view_angles[:,[which_view]],origin_position)
        n_frame = torch.nn.functional.normalize(n_frame,dim=1)
        t_frame,b_frame = torch_render.build_frame_f_z(n_frame,None,with_theta=False)
        n_hogwild = n_local[:,[0]]*t_frame+n_local[:,[1]]*b_frame+n_local[:,[2]]*n_frame
        t_hogwild = t_local[:,[0]]*t_frame+t_local[:,[1]]*b_frame+t_local[:,[2]]*n_frame
        b_hogwild = b_local[:,[0]]*t_frame+b_local[:,[1]]*b_frame+b_local[:,[2]]*n_frame
        frame_hogwild = torch_render.rotate_vector_along_axis(args.setup,-sampled_view_angles[:,[which_view]],[n_hogwild,t_hogwild,b_hogwild],is_list_input=True)
        frame_hogwild_unvisible = [-1*an_axis for an_axis in frame_hogwild]
        axay_hogwild = torch.from_numpy(mine.rejection_sampling_axay(batch_size).astype(np.float32)).to(device)
        pd_hogwild = torch.from_numpy(np.random.uniform(mine.param_bounds["pd"][0],mine.param_bounds["pd"][1],[batch_size,1]).astype(np.float32)).to(device)
        ps_hogwild = torch.from_numpy(np.random.uniform(mine.param_bounds["ps"][0],mine.param_bounds["ps"][1],[batch_size,1]).astype(np.float32)).to(device)

        disturbed_frame = origin_frame
        disturbed_param = origin_param.clone()
        disturbed_position = origin_position

        #step 1 distrub frame
        tmp_x2 = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(device)
        disturb_dir = origin_frame[1]*tmp_x2[:,[0]]+origin_frame[2]*tmp_x2[:,[1]]
        disturb_len = torch.from_numpy(np.random.normal(0.0,args.disturb_stddev["normal"],[batch_size,1]).astype(np.float32)).to(device)
        disturbed_normal = origin_frame[0]+disturb_dir*disturb_len
        disturbed_normal = torch.nn.functional.normalize(disturbed_normal,dim=1)

        tmp_x2 = torch.from_numpy(np.random.rand(batch_size,2).astype(np.float32)).to(device)
        disturb_dir = origin_frame[0]*tmp_x2[:,[0]]+origin_frame[2]*tmp_x2[:,[1]]
        disturb_len = torch.from_numpy(np.random.normal(0.0,args.disturb_stddev["tangent"],[batch_size,1]).astype(np.float32)).to(device)
        disturbed_tangent = origin_frame[1]+disturb_dir*disturb_len
        disturbed_tangent = torch.nn.functional.normalize(disturbed_tangent,dim=1)
        
        disturbed_binormal = torch.cross(disturbed_normal,disturbed_tangent)
        disturbed_tangent = torch.cross(disturbed_binormal,disturbed_normal)

        disturbed_frame = [disturbed_normal,disturbed_tangent,disturbed_binormal]

        #step 2 disturb param
        tmp_noise = torch.randn(batch_size,4,device=device)
        disturbed_param[:,3:5] = torch.where(
            visible_flag_origin,
            torch.clamp(disturbed_param[:,3:5]*torch.from_numpy(np.random.normal(1.0,args.disturb_stddev["axay_disturb"],[batch_size,2]).astype(np.float32)).to(device),mine.param_bounds["a"][0],mine.param_bounds["a"][1]),
            axay_hogwild
        )
        # disturbed_param[:,[5]] = torch.where(
        #     visible_flag_geo,
        #     torch.where(
        #         visible_flag_origin,
        #         torch.clamp(disturbed_param[:,[5]]*torch.from_numpy(np.random.normal(1.0,args.disturb_stddev["rhod"],[batch_size,1]).astype(np.float32)).to(device),mine.param_bounds["pd"][0],mine.param_bounds["pd"][1]),
        #         pd_hogwild
        #     ),
        #     torch.zeros_like(pd_hogwild)
        # )
        disturbed_param[:,[5]] = torch.where(
            visible_flag_origin,
            torch.clamp(disturbed_param[:,[5]]*torch.from_numpy(np.random.normal(1.0,args.disturb_stddev["rhod"],[batch_size,1]).astype(np.float32)).to(device),mine.param_bounds["pd"][0],mine.param_bounds["pd"][1]),
            pd_hogwild
        )
        # disturbed_param[:,[6]] = torch.where(
        #     visible_flag_geo,
        #     torch.where(
        #         visible_flag_origin,
        #         torch.clamp(disturbed_param[:,[6]]*torch.from_numpy(np.random.normal(1.0,args.disturb_stddev["rhos"],[batch_size,1]).astype(np.float32)).to(device),mine.param_bounds["ps"][0],mine.param_bounds["ps"][1]),
        #         ps_hogwild
        #     ),
        #     torch.zeros_like(ps_hogwild)
        # )
        disturbed_param[:,[6]] = torch.where(
            visible_flag_origin,
            torch.clamp(disturbed_param[:,[6]]*torch.from_numpy(np.random.normal(1.0,args.disturb_stddev["rhos"],[batch_size,1]).astype(np.float32)).to(device),mine.param_bounds["ps"][0],mine.param_bounds["ps"][1]),
            ps_hogwild
        )

        tmp_is_visible_origin = torch.where(
            visible_flag_geo,
            visible_flag_origin,
            torch.zeros_like(visible_flag_origin)
        )

        for which_axis in range(3):
            disturbed_frame[which_axis] = torch.where(
                visible_flag_geo,
                torch.where(
                    visible_flag_origin,
                    disturbed_frame[which_axis],
                    frame_hogwild[which_axis]
                ),
                frame_hogwild_unvisible[which_axis]
            )
        
        global_frame_noised_list.append(disturbed_frame)
        input_params_noised_list.append(disturbed_param)
        input_positions_noised_list.append(disturbed_position)
        is_visible_origin_flag_list.append(tmp_is_visible_origin)
    
    is_visible_origin_flag_tensor = torch.cat(is_visible_origin_flag_list,dim=1)#(batch,sample_view_num)

    visibility_collector = []
    loss_weight_collector_shading = []
    loss_weight_collector_geo = []
    visibility_collector_gt = []#a
    visibility_collector_geo = []#a
    for which_view in range(args.sample_view_num):
        tmp_wo_dot_n = torch_render.compute_wo_dot_n(args.setup,input_positions_noised_list[which_view],sampled_view_angles[:,[which_view]],global_frame_noised_list[which_view][0],args.setup.get_cam_pos_torch(device))
        visibility_collector.append(tmp_wo_dot_n)

        normal_local_view = torch_render.rotate_vector_along_axis(args.setup,sampled_view_angles[:,[which_view]],global_frame[0])
        # normal_local_view[:,2] = 0.0
        normal_geo_local_view = torch_render.rotate_vector_along_axis(args.setup,sampled_view_angles[:,[which_view]],normal_geo)
        # normal_geo_local_view[:,2] = 0.0
        normal_local_view = torch.nn.functional.normalize(normal_local_view,dim=1)
        normal_geo_local_view = torch.nn.functional.normalize(normal_geo_local_view,dim=1)
        position_local_view = torch_render.rotate_point_along_axis(args.setup,sampled_view_angles[:,[which_view]],input_positions)
        wo_local_view = args.setup.get_cam_pos_torch(device) - position_local_view#(batch,3)
        # wo_local_view_reduced = wo_local_view.clone()
        # wo_local_view_reduced[:,2] = 0.0
        # wo_local_view_reduced = torch.nn.functional.normalize(wo_local_view_reduced,dim=1)
        wo_local_view = torch.nn.functional.normalize(wo_local_view,dim=1)
        dot_value = torch.sum(normal_local_view*wo_local_view,dim=1,keepdim=True)#(batch,1)
        dot_value_geo = torch.sum(normal_geo_local_view*wo_local_view,dim=1,keepdim=True)#(batch,1)
        decay_theta = 2.0*math.pi/args.sample_view_num#*0.5
        loss_weight = dot_value
        loss_weight_geo = dot_value_geo#compute_loss_weight(dot_value_geo,args.sample_view_num)
        loss_weight_collector_shading.append(loss_weight)
        loss_weight_collector_geo.append(loss_weight_geo)

        if args.visualize:
            tmp_wo_dot_n = torch_render.compute_wo_dot_n(args.setup,input_positions,sampled_view_angles[:,[which_view]],global_frame[0],args.setup.get_cam_pos_torch(device))#a
            visibility_collector_gt.append(tmp_wo_dot_n)#a
            tmp_wo_dot_n = torch_render.compute_wo_dot_n(args.setup,input_positions,sampled_view_angles[:,[which_view]],normal_geo,args.setup.get_cam_pos_torch(device))#a
            visibility_collector_geo.append(tmp_wo_dot_n)

    visibility_tensor = torch.cat(visibility_collector,dim=1)#(batch,sample_view_num)
    visibility_tensor = visibility_tensor > 0.0

    loss_weight_tensor_geo = torch.cat(loss_weight_collector_geo,dim=1)#(batch,sample_view_num)
    loss_weight_tensor_shading = torch.cat(loss_weight_collector_shading,dim=1)#(batch,sample_view_num)

    valid_ratio = torch.sum(is_visible_origin_flag_tensor,dim=1,keepdim=True).to(dtype=torch.float32)/torch.sum(visibility_tensor,dim=1,keepdim=True).to(dtype=torch.float32)#(batch,1)

    if args.visualize:
        visibility_tensor_gt = torch.cat(visibility_collector_gt,dim=1)#a
        visibility_tensor_gt = visibility_tensor_gt > 0.0#a
        visibility_tensor_geo = torch.cat(visibility_collector_geo,dim=1)#a
        visibility_tensor_geo = visibility_tensor_geo > 0.0#a

    if args.visualize:
        return loss_weight_tensor_geo,loss_weight_tensor_shading,valid_ratio,normal_geo,input_params,input_positions,input_params_noised_list,global_frame_noised_list,input_positions_noised_list,global_frame,visibility_tensor,visibility_tensor_gt,visibility_tensor_geo
    else:
        return loss_weight_tensor_geo,loss_weight_tensor_shading,valid_ratio,normal_geo,input_params,input_positions,input_params_noised_list,global_frame_noised_list,input_positions_noised_list,global_frame,visibility_tensor

def generate_data(target_num,pf_output,device,args,pf_gt_param,batch_size = 15000):
    counter = 0
     
    sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=args.sample_view_num,endpoint=False)#[self.sample_view_num]
    sampled_rotate_angles_np = (np.expand_dims(sampled_rotate_angles_np,axis=0)).astype(np.float32)
    
    while True:
        # print("generating a batch-----")
        if counter == target_num:
            break
        assert counter < target_num,"error occured!"
        if counter % 1000 == 0:
            print("counter/total:{}/{}".format(counter,target_num))
        cur_batch_size = target_num-counter if target_num-counter < batch_size else batch_size
        wanted_num = cur_batch_size
        
        first_itr = True
        input_params_collector = None
        input_positions_collector = None
        geometry_normal_collector = None
        global_frame_collector = []
        input_params_noised_list_collector = []
        global_frame_noised_list_collector = []
        input_positions_noised_list_collector = []
        visibility_gt = None#a
        visibility_geo = None#a
        visibility_tensor_collector = None
        loss_weight_tensor_collector_shading = None
        loss_weight_tensor_collector_geo = None
    
        while True:
            # print("wanted num:{}".format(wanted_num))
            if args.visualize:
                loss_weight_tensor_geo,loss_weight_tensor_shading,valid_ratio,normal_geo,input_params,input_positions,input_params_noised_list,global_frame_noised_list,input_positions_noised_list,global_frame,visibility_tensor,visibility_tensor_gt,visibility_tensor_geo = generate_batch_data(wanted_num,device,args)#a
            else:
                loss_weight_tensor_geo,loss_weight_tensor_shading,valid_ratio,normal_geo,input_params,input_positions,input_params_noised_list,global_frame_noised_list,input_positions_noised_list,global_frame,visibility_tensor = generate_batch_data(wanted_num,device,args)
            #check_valid here
            loss_weight_tensor_geo_real = compute_loss_weight(loss_weight_tensor_geo,args.sample_view_num)
            loss_weight_tensor_shading_real = compute_loss_weight(loss_weight_tensor_shading,args.sample_view_num)
            visibile_view_num = torch.sum(visibility_tensor,dim=1,keepdim=True)#(batch,1)
            sw_geo_valid_num = torch.sum(loss_weight_tensor_geo_real > 0.0,dim=1,keepdim=True)
            sw_shading_valid_num = torch.sum(loss_weight_tensor_shading_real > 0.0,dim=1,keepdim=True)
            geo_best_view_id = torch.argmax(loss_weight_tensor_geo,dim=1)#(batch,)
            label_visible_flag = torch.unsqueeze(loss_weight_tensor_shading[np.arange(wanted_num),geo_best_view_id] > 0.0,dim=1)#(batch,1)
            # valid_sample_mask = torch.where(torch.ones_like(visibile_view_num > 0))[0]#no rule
            # valid_sample_mask = torch.where((visibile_view_num > 0) & (sw_geo_valid_num > 0) & (sw_shading_valid_num > 0))[0]#rule 1
            valid_sample_mask = torch.where((visibile_view_num > 0) & (valid_ratio > 0.5) & (sw_geo_valid_num > 0) & (sw_shading_valid_num > 0) & label_visible_flag)[0]
            valid_num = len(valid_sample_mask)
            # print("valid_num:{}".format(valid_num))
            #collect valid samples to wallet
            # loss_weight_tensor = torch.nn.functional.normalize(loss_weight_tensor,dim=1)
            if first_itr:
                loss_weight_tensor_collector_shading = loss_weight_tensor_shading[valid_sample_mask]
                loss_weight_tensor_collector_geo = loss_weight_tensor_geo[valid_sample_mask]
                visibility_tensor_collector = visibility_tensor[valid_sample_mask]
                geometry_normal_collector = normal_geo[valid_sample_mask]
                input_params_collector = input_params[valid_sample_mask]
                input_positions_collector = input_positions[valid_sample_mask]
                for an_axis in global_frame:
                    global_frame_collector.append(an_axis[valid_sample_mask])
                for a_item in input_params_noised_list:
                    input_params_noised_list_collector.append(a_item[valid_sample_mask])
                for a_item in global_frame_noised_list:
                    global_frame_noised_list_collector.append([an_axis[valid_sample_mask] for an_axis in a_item])
                for a_item in input_positions_noised_list:
                    input_positions_noised_list_collector.append(a_item[valid_sample_mask])
                if args.visualize:
                    visibility_gt = visibility_tensor_gt[valid_sample_mask]#a
                    visibility_geo = visibility_tensor_geo[valid_sample_mask]#a
                first_itr = False
            else:
                loss_weight_tensor_collector_shading = torch.cat([loss_weight_tensor_collector_shading,loss_weight_tensor_shading[valid_sample_mask]],dim=0)
                loss_weight_tensor_collector_geo = torch.cat([loss_weight_tensor_collector_geo,loss_weight_tensor_geo[valid_sample_mask]],dim=0)
                visibility_tensor_collector = torch.cat([visibility_tensor_collector,visibility_tensor[valid_sample_mask]],dim=0)
                geometry_normal_collector = torch.cat([geometry_normal_collector,normal_geo[valid_sample_mask]],dim=0)
                input_params_collector = torch.cat([input_params_collector,input_params[valid_sample_mask]],dim=0)
                input_positions_collector = torch.cat([input_positions_collector,input_positions[valid_sample_mask]],dim=0)
                for idx,an_axis in enumerate(global_frame):
                    global_frame_collector[idx] = torch.cat([global_frame_collector[idx],an_axis[valid_sample_mask]],dim=0)
                for idx,a_item in enumerate(input_params_noised_list):
                   input_params_noised_list_collector[idx] = torch.cat([input_params_noised_list_collector[idx],a_item[valid_sample_mask]],dim=0)
                for which_view in range(args.sample_view_num):
                    for idx in range(3):
                        global_frame_noised_list_collector[which_view][idx] = torch.cat([global_frame_noised_list_collector[which_view][idx],global_frame_noised_list[which_view][idx][valid_sample_mask]],dim=0)
                for idx,a_item in enumerate(input_positions_noised_list):
                    input_positions_noised_list_collector[idx] = torch.cat([input_positions_noised_list_collector[idx],a_item[valid_sample_mask]],dim=0)
                if args.visualize:
                    visibility_gt = torch.cat([visibility_gt,visibility_tensor_gt[valid_sample_mask]],dim=0)#a
                    visibility_geo = torch.cat([visibility_geo,visibility_tensor_geo[valid_sample_mask]],dim=0)#a
            wanted_num -= valid_num
            if wanted_num == 0:
                break
        counter += cur_batch_size

        if args.rendering:
            sampled_rotate_angles = np.repeat(sampled_rotate_angles_np,cur_batch_size,axis=0)#[batch,sample_view_num]
            sampled_view_angles = torch.from_numpy(sampled_rotate_angles).to(device)
            ### rendering input lumi
            multiview_lumitexel_list,_ = args.multiview_renderer(input_params_noised_list_collector,input_positions_noised_list_collector,sampled_view_angles,global_frame=global_frame_noised_list_collector)#a list item shape=(batchsize,lumilen,channel_num)
            multiview_lumitexel_list = [a_lumi * args.RENDER_SCALAR for a_lumi in multiview_lumitexel_list]#a list item shape=(batchsize,lumilen,channel_num)
            
            ### rendering label
            multiview_lumitexel_gt_list,_ = args.multiview_renderer_gt(input_params_collector,input_positions_collector,sampled_view_angles,global_frame=global_frame_collector)#a list item shape=(batchsize,lumilen,channel_num)
            multiview_lumitexel_gt_list = [a_lumi * args.RENDER_SCALAR for a_lumi in multiview_lumitexel_gt_list]#a list item shape=(batchsize,lumilen,channel_num)

        #output here
        if args.projection:
            m_noised_list = []
            m_gt_list = []
            for which_view in range(args.sample_view_num):
                tmpW = args.W_list[which_view].to(multiview_lumitexel_list[which_view].device).repeat(cur_batch_size,1,1)
                tmp_m = torch.matmul(tmpW,multiview_lumitexel_list[which_view])#(batch,m_len,1)
                tmp_m = tmp_m*(torch.from_numpy(np.random.normal(1.0,0.05,[cur_batch_size,args.m_len,1]).astype(np.float32)).to(tmp_m.device))#(torch.randn_like(tmp_m)*0.05+1.0)
                m_noised_list.append(tmp_m)

                tmp_m = torch.matmul(tmpW,multiview_lumitexel_gt_list[which_view])#(batch,m_len,1)
                m_gt_list.append(tmp_m)
        
            m_noised_tensor = torch.stack(m_noised_list,dim=1)#(batch,sampleviewnum,m_len,1)
            m_gt_tensor = torch.stack(m_gt_list,dim=1)#(batch,sampleviewnum,m_len,1)

            m_tensor = torch.stack([m_noised_tensor,m_gt_tensor],dim=1)#(batch,2,sampleviewnum,m_len,1)
            
            m_tensor_np = m_tensor.cpu().numpy()#(batch,2,sampleviewnum,m_len,1)
            geometry_normal_collector = geometry_normal_collector.cpu().numpy()#(batch,3)

            data_set = np.concatenate([np.reshape(m_tensor_np,[m_tensor_np.shape[0],-1]),geometry_normal_collector],axis=-1)
            data_set.astype(np.float32).tofile(pf_output)
        else:
            #selecte randomely
            selected_branch_random = (np.random.rand(cur_batch_size,1)*args.sample_view_num).astype(np.float32)#(batch,1)
            selected_branch_random = torch.from_numpy(selected_branch_random).to(device)
            #select visible view
            visible_flag = np.logical_and(loss_weight_tensor_collector_shading.cpu().numpy()>0.0,loss_weight_tensor_collector_geo.cpu().numpy()>0.0)
            viewed_num = np.sum(visible_flag,axis=1)
            selected_idx_row = (np.random.rand(cur_batch_size)*viewed_num.astype(np.float32)).astype(np.int32)#(batch_size,)
            first_invisible_idx = np.argmin(visible_flag,axis=1)
            first_visible_idx = np.argmax(visible_flag,axis=1)

            adjust_idx = (first_invisible_idx+args.sample_view_num-viewed_num)%args.sample_view_num
            first_visible_idx =np.where(first_visible_idx==0,
                adjust_idx,
                first_visible_idx
            )
            random_visible_idx = (selected_idx_row+first_visible_idx)%args.sample_view_num
            selected_branch_visible_io = (np.reshape(random_visible_idx,[-1,1])).astype(np.float32)#(batch_size,1)
            selected_branch_visible_io = torch.from_numpy(selected_branch_visible_io).to(device)

            #assemble params here
            param_list = [
                input_params_collector,#7
                input_positions_collector,#3
                global_frame_collector[0],#3
                global_frame_collector[1],#3
                global_frame_collector[2]#3
            ]
            for which_view in range(args.sample_view_num):
                tmp_param_list = [
                    input_params_noised_list_collector[which_view],#7
                    input_positions_noised_list_collector[which_view],#3
                    global_frame_noised_list_collector[which_view][0],#3
                    global_frame_noised_list_collector[which_view][1],#3
                    global_frame_noised_list_collector[which_view][2]#3
                ]
                param_list.extend(tmp_param_list)

            param_list.append(geometry_normal_collector)    
            param_list.append(loss_weight_tensor_collector_geo)
            param_list.append(loss_weight_tensor_collector_shading)
            param_list.append(selected_branch_random)
            param_list.append(selected_branch_visible_io)

            param_tensor = torch.cat(
                param_list,dim=-1
            )
            param_tensor = param_tensor.cpu().numpy()
            param_tensor.astype(np.float32).tofile(pf_output)

        ################a
        if args.visualize:
            assert args.rendering and args.projection,"What the heck do you want to visualize?"
            tensor_block_size = 20
            tensor_height_resized = args.sample_view_num*tensor_block_size
            tensor_width_resized = args.m_len*tensor_block_size

            multiview_lumitexel_tensor = torch.stack(multiview_lumitexel_list,dim=1)#(batchsize,sampleviewnum,lumilen,channelnum)
            multiview_lumitexel_gt_tensor = torch.stack(multiview_lumitexel_gt_list,dim=1)
            lumi_img_height = 64*3
            lumi_img_width = 64*4
            lumi_img_total_len = lumi_img_height*lumi_img_width
            for which_sample in range(cur_batch_size):
                # if counter-cur_batch_size+which_sample == 125:
                #     print("gt visibility")
                #     print(visibility_gt[which_sample])
                #     print("geo visibility")
                #     print(visibility_geo[which_sample])
                #     print("sample visibility")
                #     print(visibility_tensor_collector[which_sample])
                #     print("param disturbed")
                #     print(input_params_noised_list_collector[6][which_sample])
                #     print("param input")
                #     print(input_params_collector[which_sample])
                # if counter-cur_batch_size+which_sample == 126:
                #     time.sleep(5)
                #     exit()
                
                img_stack_input = torch_render.visualize_lumi(multiview_lumitexel_tensor[which_sample].cpu().numpy(),args.setup)
                img_stack_gt = torch_render.visualize_lumi(multiview_lumitexel_gt_tensor[which_sample].cpu().numpy(),args.setup)
                img_grid_input = torchvision.utils.make_grid(torch.from_numpy(img_stack_input).permute(0,3,1,2),nrow=2, pad_value=0.5)#(channel,height,width)
                img_grid_gt = torchvision.utils.make_grid(torch.from_numpy(img_stack_gt).permute(0,3,1,2),nrow=2, pad_value=0.5)#(channel,height,width)
                tmp_v_gt = visibility_gt[which_sample].to(dtype=torch.float32).reshape(8,1,1).repeat(1,lumi_img_total_len,3)
                tmp_v_geo = visibility_geo[which_sample].to(dtype=torch.float32).reshape(8,1,1).repeat(1,lumi_img_total_len,3)
                img_grid_v_gt = torchvision.utils.make_grid(tmp_v_gt.reshape(8,lumi_img_height,lumi_img_width,3).permute(0,3,1,2),nrow=2, pad_value=0.5)
                img_grid_v_geo = torchvision.utils.make_grid(tmp_v_geo.reshape(8,lumi_img_height,lumi_img_width,3).permute(0,3,1,2),nrow=2, pad_value=0.5)

                img_grid = torch.cat([img_grid_input,img_grid_gt,img_grid_v_geo.cpu(),img_grid_v_gt.cpu()],dim=2)
                img_grid = torch.clamp(img_grid,0.0,1.0)

                tmp_m_tensor = m_tensor[which_sample]
                m_gt_max = torch.abs(tmp_m_tensor[1]).max()
                tensor_list = []
                for tmp_tensor in tmp_m_tensor:
                    tmp_tensor = tmp_tensor/(m_gt_max+1e-6)
                    tmp_tensor = torch.cat([torch.clamp(tmp_tensor,min=0.0),torch.zeros_like(tmp_tensor),torch.clamp(tmp_tensor,max=0.0)*-1.0],dim=2)
                    tmp_tensor = torch.unsqueeze(torch.unsqueeze(tmp_tensor,dim=2),dim=1)#(sample_view,1,m_len,1,1) torch tensor
                    tmp_tensor = tmp_tensor.repeat(1,tensor_block_size,1,tensor_block_size,1).reshape(tensor_height_resized,tensor_width_resized,3)
                    tmp_tensor = tmp_tensor.permute(2,0,1)
                    tensor_list.append(tmp_tensor)
                tensor_list = torch.stack(tensor_list,dim=0)#(id,channel,height,width)
                tensor_list = torch.clamp(tensor_list,0.0,1.0)
                tmp_tensor_img = torchvision.utils.make_grid(tensor_list,nrow=3, pad_value=0.5)
                tmp_tensor_img = torch.cat([
                    tmp_tensor_img,
                    torch.ones(3,tmp_tensor_img.size()[1],img_grid.size()[2]-tmp_tensor_img.size()[2],device=tmp_tensor_img.device)*0.5
                ],dim=2)

                img_grid = torch.cat(
                    [
                        img_grid,
                        tmp_tensor_img.to(img_grid.device)
                    ],
                    dim=1
                )

                args.writer.add_image(
                    "lumi/{}".format(counter-cur_batch_size+which_sample), 
                    img_grid,  
                    global_step=0, dataformats='CHW')


if __name__ == "__main__":
    np.random.seed(233)
    random.seed(233)
    torch.random.manual_seed(233)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="../../training_data/")
    parser.add_argument("--data_set_name",default="noised_param_20m_sig20like_12view")
    parser.add_argument("--total_num",type=int,default=20000000)
    parser.add_argument("--train_ratio",type=float,default=0.9)
    parser.add_argument("--sample_view_num",type=int,default=12)
    parser.add_argument("--m_len",type=int,default=3)
    parser.add_argument("--visualize",action="store_true")
    parser.add_argument("--save_gt_param",action="store_true")
    parser.add_argument("--rendering",action="store_true")
    parser.add_argument("--projection",action="store_true")
    parser.add_argument("--pretrained_W_path",type=str,default="/home/cocoa_kang/training_tasks/current_work/newborn_torch/trained_models/material_net/first_blood/models/")

    args = parser.parse_args()

    args.disturb_stddev = {}
    args.disturb_stddev["geo_normal"] = 0.5
    args.disturb_stddev["normal"] = 0.0
    args.disturb_stddev["tangent"] = 0.0
    args.disturb_stddev["axay_disturb"] = 0.15
    args.disturb_stddev["rhod"] = 0.1
    args.disturb_stddev["rhos"] = 0.1

    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
    setup_input = Setup_Config(standard_rendering_parameters)
    args.setup = setup_input

    #########################################################
    save_root = args.data_root+args.data_set_name+"/"
    os.makedirs(save_root,exist_ok=True)
    train_data_num = int(args.total_num*args.train_ratio)
    val_data_num = args.total_num - train_data_num

    #########################################################
    if args.rendering:
        multiview_render_args = {
            "available_devices":[torch.device("cuda:{}".format(i)) for i in range(torch.device_count())],#,torch.device("cuda:3")],
            "torch_render_path":TORCH_RENDER_PATH,
            "rendering_view_num":args.sample_view_num,
            "setup":args.setup,
            "renderer_name_base":"multiview_renderer",
            "renderer_configs":["ntb"],
            "input_as_list":True
        }
        multiview_renderer = Multiview_Renderer(multiview_render_args,max_process_live_per_gpu=5)
        multiview_render_args = multiview_render_args.copy()
        multiview_render_args["renderer_name_base"] = "multiview_renderer_gt"
        multiview_render_args["input_as_list"] = False
        multiview_renderer_gt = Multiview_Renderer(multiview_render_args,max_process_live_per_gpu=5)

        args.multiview_renderer = multiview_renderer
        args.multiview_renderer_gt = multiview_renderer_gt
        args.RENDER_SCALAR = 5*1e3/math.pi

    if args.visualize:
        assert args.rendering == True and args.projection == True,"What do you want to visualize?"
        writer = SummaryWriter()#a
        args.writer = writer#a

    if args.projection:
        args.W_list = []
        for which_view in range(args.sample_view_num):
            tmpW = np.fromfile(args.pretrained_W_path+"{}/W.bin".format(which_view),np.float32).reshape([args.m_len,args.setup.get_light_num(),3])
            tmpW = tmpW[:,:,0]
            tmpW = torch.from_numpy(tmpW)
            tmpW = torch.unsqueeze(tmpW,dim=0)#[batchsize,measurementnum,lightnum]
        
            args.W_list.append(tmpW)
    ########################################################
    gen_device = torch.device("cuda:0")
    print("generating train data...")
    pf_gt_param = open(save_root+"train_gt_param.bin","wb") if args.save_gt_param else None
    with open(save_root+"train.bin","wb") as pf:
        generate_data(train_data_num,pf,gen_device,args,pf_gt_param)
    print("done.")
    if args.visualize or args.save_gt_param:
        time.sleep(5.0)
        exit()
    print("generating val data....")
    pf_gt_param = open(save_root+"val_gt_param.bin","wb") if args.save_gt_param else None
    with open(save_root+"val.bin","wb") as pf:
        generate_data(val_data_num,pf,gen_device,args,pf_gt_param)
    print("done.")
    
