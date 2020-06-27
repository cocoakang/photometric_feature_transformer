import torch
import numpy as np
import sys
sys.path.append("../torch_renderer/")
import torch_render
import os
import open3d as o3d
import math
import matplotlib.pyplot as plt
def compute_loss_weight(normal,wo,decay_range,concentrate_ratio=1.0):
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
    
    in_north = math.pi-max1_end > decay_range
    
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
    return loss_weight,max1_end,in_north

def compute_loss_weight_old(normal,wo,decay_range):
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
    alpha = torch.where(normal[:,[2]] > 0.0,alpha,2*math.pi-alpha)
    alpha = torch.where(wo[:,[2]] < 0.0,2.0*math.pi - alpha,alpha)
    alpha = alpha*0.5
    end_theta = math.pi-alpha#torch.where(wo[:,[2]] < 0.0,end_theta_origin,math.pi - end_theta_origin)
    # test = end_theta
    # print("end_theta:",end_theta.reshape(-1,50)[:30,-1]/math.pi*180.0)

    # max1_start = 0.0
    max1_end = torch.clamp(end_theta-decay_range,min=0.0)
    # print("max1_end:",max1_end/math.pi*180.0)

    B = torch.cos(math.pi*0.5-torch.acos(torch.abs(wo[:,[2]])))#nz_bound
    # print("B:",torch.acos(B)/math.pi*180.0)
    # print("normal:",torch.acos(torch.abs(normal[:,[2]]))/math.pi*180.0)
    shift = decay_range/(1-B)*(torch.abs(normal[:,[2]])-B)     
    # print("shift:",shift/math.pi*180.0)
    shift = torch.where(torch.abs(normal[:,[2]]) == 1.0,torch.ones_like(shift).fill_(decay_range),shift)
    shift = torch.where(end_theta < (math.pi-1e-4),torch.zeros_like(shift),shift)# 
    # print("shift:",shift.reshape(-1,50)[:14,-1]/math.pi*180.0)

    max1_end = max1_end+shift
    # max1_end = torch.where(
    #     end_theta > (math.pi),# - 1e-4
    #     torch.clamp(max1_end+shift,max=math.pi),
    #     max1_end
    # )
    print(math.pi-max1_end)
    in_north = math.pi-max1_end > decay_range
    print(in_north)


    # print("beta:",beta)
    normal_xy = normal.clone()
    normal_xy[:,2] = 0.0
    normal_xy = torch.nn.functional.normalize(normal_xy)
    wo_xy = wo.clone()
    wo_xy[:,2] = 0.0
    wo_xy = torch.nn.functional.normalize(wo_xy)
    theta = torch.acos(torch.clamp(torch.sum(normal_xy*wo_xy,dim=1,keepdims=True),min=-1.0,max=1.0))

    # loss_weight = (end_theta-theta+shift)/decay_range
    loss_weight = 0.5*torch.cos(-math.pi/decay_range*theta+math.pi/decay_range*(end_theta-decay_range+shift))+0.5#(end_theta-theta+shift)/decay_range

    loss_weight = torch.where(theta <= max1_end,torch.ones_like(loss_weight),loss_weight)
    loss_weight = torch.where(theta > end_theta ,torch.zeros_like(loss_weight),loss_weight)
    
    # loss_weight = torch.where(loss_weight > 0.0 ,torch.ones_like(loss_weight),torch.zeros_like(loss_weight))
    # loss_weight = torch.where(
    #     end_theta >= math.pi-1e-4,
    #     torch.clamp(loss_weight+decay_range/(1-B)*shift,max=1.0),
    #     loss_weight
    # )

    # print("loss_weight:",loss_weight)
    return loss_weight,max1_end,in_north


np.random.seed(667)

wo = torch.from_numpy(np.array([[0.0,1.0,1.0]],np.float32))#(1,3)
# normal = torch.from_numpy(np.array([[0.0,1.0,-0.8]],np.float32))#(1,3)
# normal = torch.from_numpy(np.array([[0.0,math.sqrt(3)*0.5,0.5]],np.float32))#(1,3)
wo = torch.nn.functional.normalize(wo,dim=1)#(1,3)

sample_view_num = 8

height_num = 100
row_num = 50
height = np.linspace(1.0,-1.0,num=height_num)#(height_num,)
r = np.sqrt(1.0-np.square(np.abs(height)))#(height_num,)

row_theta = np.linspace(-math.pi,math.pi,row_num)#()
xy = np.stack([np.sin(row_theta),np.cos(row_theta)],axis=1)#(row_num,2)
xy = np.repeat(np.expand_dims(xy,axis=0),height_num,axis=0)#(heightnum,rownum,2)
xy = xy*np.reshape(r,[height_num,1,1])
z = np.repeat(np.reshape(height,[height_num,1,1]),row_num,axis=1)
data = np.concatenate([xy,z],axis=2)
normal = torch.from_numpy(np.reshape(data,[-1,3]).astype(np.float32))
sample_num = normal.size()[0]

# sample_num = 100000
# data = torch.from_numpy(np.random.uniform(0.0,1.0,[sample_num,2])).float()
# wo = torch_render.back_full_octa_map(data)
# wo[:,2] = 1.0

# sample_num = 1
# # wo = torch.from_numpy(np.array([[0.0,-0.0001,1.0]],np.float32))#torch_render.back_full_octa_map(data)
# wo = normal.clone()
# wo[:,2] = wo[:,2]-0.0001


normal = torch.nn.functional.normalize(normal,dim=1)

decay_theta_old = 2.0*math.pi/sample_view_num
decay_theta = 2.0*math.pi/sample_view_num

tmp_wo = wo.repeat(sample_num,1)

loss_weight,max1_end,in_north = compute_loss_weight(normal,tmp_wo,decay_theta)
loss_weight_old,max1_end_old,in_north = compute_loss_weight_old(normal,tmp_wo,decay_theta_old)

pcd = o3d.geometry.PointCloud()

wo_num = 50

#draw normalplane
wo_plane_num = 1000
tmp_points = np.random.rand(wo_plane_num,3)*2.0-1.0
wo_cpu = wo.cpu().numpy()#(1,3)
wo_plane = np.cross(wo_cpu,np.cross(wo_cpu,tmp_points))
wo_plane = wo_plane/np.linalg.norm(wo_plane,axis=1,keepdims=True)

points = np.concatenate(
    [
        normal,
        wo.cpu().numpy()*np.expand_dims(np.linspace(0.0,2.0,num=wo_num),axis=1),
        wo_plane
    ],axis=0
)
colors = np.concatenate(
    [
        loss_weight.repeat(1,3).numpy(),
        np.repeat(np.array([[1.0,0.0,0.0]]),wo_num,axis=0),
        np.repeat(np.array([[0.0,1.0,0.0]]),wo_plane_num,axis=0)
    ],axis=0
)
save_root = "sw/"
os.makedirs(save_root,exist_ok=True)

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(save_root+"sw_output.ply", pcd)


loss_weight = np.reshape(loss_weight,[height_num,row_num])
loss_weight_old = np.reshape(loss_weight_old,[height_num,row_num])
fig = plt.figure()


for which_row in range(height_num):
    plt.title("height={}".format(height[which_row]))
    plt.xlabel("angle degree between wo and norml in equatorial plane")
    plt.ylabel("weight")
    plt.ylim(0.0,1.0)
    plt.xlim(-180.0,180.0)
    plt.scatter(row_theta/math.pi*180.0,loss_weight[which_row],color='r')
    plt.scatter(row_theta/math.pi*180.0,loss_weight_old[which_row],color='b')
    plt.savefig(save_root+"{}.png".format(which_row))
    plt.clf()

theta_height = -torch.acos(normal[:,2]).numpy()
plt.scatter(theta_height/math.pi*180.0+90.0,max1_end/math.pi*180.0,color='r')
plt.scatter(theta_height/math.pi*180.0+90.0,max1_end_old/math.pi*180.0,color='b')
plt.title("normalheight-max1theta")
plt.xlabel("height")
plt.ylim(0.0,180.0)
plt.ylabel("max1theta")
plt.savefig(save_root+"height-max1theta.png")