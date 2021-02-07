import torch
import torchvision
import numpy as np
import sys
import shutil
import os
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from sklearn.decomposition import PCA
import cv2
import math
import struct
from subprocess import Popen

class DIFT_QUALITY_CHECKER_DILI:
    '''
    This class is used to test trained dift model. 
    Specifically, it aims at showing the pattern of a trained dift feature instead of loss.
    It takes in some metadata of an object, then reder it into lumitexels which are transfered to dift later.
    The infered dift codes will be projected to visualizable lengths(3).
    '''

    def __init__(self,training_configs,log_dir,metadata_root,checker_name,test_device,axay=None,diff_albedo=None,spec_albedo=None,batch_size=1000,test_view_num=1,test_in_grey=True,check_type="a"):
        '''
        metadata_root: the folder contains exrs(pos.exr normal.exr (tangent.exr axay.exr pd.exr ps.exr maybe))
        check_type= g or m or a
        '''
        ########################################
        ##loading setup configuration        ###
        ########################################
        self.setup = training_configs["setup_input"]
        self.check_type = check_type
        self.dift_code_len = training_configs["dift_code_len"] if check_type == "a" else training_configs["dift_code_config"][check_type][0]
        self.checker_name = checker_name
        self.batch_size = batch_size
        self.test_device = test_device
        self.rotate_num = test_view_num
        self.log_dir = log_dir+"/"+self.checker_name+"/"
        self.test_in_grey = test_in_grey
        os.makedirs(self.log_dir,exist_ok=True)

        self.sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=training_configs["sample_view_num_whentest"],endpoint=False)
        self.sampled_rotate_angles_np = np.expand_dims(self.sampled_rotate_angles_np,axis=0).astype(np.float32)#(1,sampleviewnum)

        self.RENDER_SCALAR = training_configs["RENDER_SCALAR"]
        ###################
        ###step 1 read in all meta exrs
        ###################
        pos_exr = cv2.imread(metadata_root+"pos.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
        normal_exr = cv2.imread(metadata_root+"normal.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
        
        files = os.listdir(metadata_root)
        tangent_presented = False
        if "tangent_fitted_global.exr" in files:
            tangent_exr = cv2.imread(metadata_root+"tangent_fitted_global.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
            tangent_presented = True

        if axay is None:
            axay_exr = cv2.imread(metadata_root+"axay_fitted.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
            axay_exr = axay_exr[:,:,:2]#(height,width,2)
        else:
            axay_exr = np.ones_like(pos_exr)[:,:,::2]
            axay_exr[:,:,0] = axay[0]
            axay_exr[:,:,1] = axay[1]

        if diff_albedo is None:
            diff_exr = cv2.imread(metadata_root+"pd_fitted.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
        else:
            diff_exr = np.ones_like(pos_exr)*diff_albedo

        if spec_albedo is None:
            spec_exr = cv2.imread(metadata_root+"ps_fitted.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,::-1]#(height,width,3)
        else:
            spec_exr = np.ones_like(pos_exr)*spec_albedo


        mask_exr = cv2.imread(metadata_root+"mask.exr",cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,0]#(height,width)
        self.valid_pixels = np.where(mask_exr > 0.99)#[(valid_num,),(valid_num,)]
        self.img_height,self.img_width = mask_exr.shape
        
        ####################
        ###step 2 reorder the exr data as array
        ####################
        self.position = pos_exr[self.valid_pixels].astype(np.float32)#(validnum,3)
        self.normal = normal_exr[self.valid_pixels].astype(np.float32)#(validnum,3)
        self.axay = axay_exr[self.valid_pixels].astype(np.float32)#(validnum,2)
        if self.test_in_grey:
            self.pd = np.mean(diff_exr[self.valid_pixels],axis=1,keepdims=True).astype(np.float32)#(validnum,1)
            self.ps = np.mean(spec_exr[self.valid_pixels],axis=1,keepdims=True).astype(np.float32)#(validnum,1)
        else:
            self.pd = diff_exr[self.valid_pixels].astype(np.float32)#(validnum,3)
            self.ps = spec_exr[self.valid_pixels].astype(np.float32)#(validnum,3)    

        self.normal = self.normal/(np.linalg.norm(self.normal,axis=1,keepdims=True)+1e-5)

        if not tangent_presented:
            '''
            tangent isn't presented, determine randomly by myself
            '''
            print("[DIFT QUALITY CHECKER] warning: tangent is generated automaticlly and randomly")
            normal_tc = torch.from_numpy(self.normal)#(pointnum,3)
            theta = np.random.uniform(0.0,math.pi,(normal_tc.shape[0],1)).astype(np.float32)
            tangent_tc,binormal_tc = torch_render.build_frame_f_z(normal_tc,torch.from_numpy(theta).to(normal_tc.device))
            self.tangent = tangent_tc.cpu().numpy()
            self.binormal = binormal_tc.cpu().numpy()
        else:
            '''
            just generate binormal using cross product
            '''
            self.tangent = tangent_exr[self.valid_pixels].astype(np.float32)#(validnum,3)
            self.tangent = self.tangent/(np.linalg.norm(self.tangent,axis=1,keepdims=True)+1e-5)
            self.binormal = np.cross(self.normal,self.tangent)

        ####################
        ###step 3 generate fake cam00_index_nocc.bin to cheat the stupid compact_dift_codes.py
        ####################
        self.valid_pixels_num = self.valid_pixels[0].shape[0]
        self.fake_index = np.stack(self.valid_pixels[::-1],axis=1).astype(np.int32)#(validnum,2),xy


    def check_quality(self,dift_trainer,writer,global_step):
        """
        dift_trainer: To get current model
        writer: tensorboard logger
        global_step: 

        TODO maybe this can be done in background?
        """
        ##############################################################################
        ## infer diftcodes
        ##############################################################################
        dift_trainer.to(self.test_device)
        
        for which_view in range(self.rotate_num):
            cur_save_root = self.log_dir+"{}/".format(which_view)
            os.makedirs(cur_save_root,exist_ok=True)
            pf_save = open(cur_save_root+"feature.bin".format(which_view),"wb")
        
            ptr = 0
            while True:
                tmp_pos = self.position[ptr:ptr+self.batch_size]
                tmp_normal = self.normal[ptr:ptr+self.batch_size]
                tmp_tangent = self.tangent[ptr:ptr+self.batch_size]
                tmp_binormal = self.binormal[ptr:ptr+self.batch_size]
                tmp_axay = self.axay[ptr:ptr+self.batch_size]
                tmp_pd = self.pd[ptr:ptr+self.batch_size]
                tmp_ps = self.ps[ptr:ptr+self.batch_size]

                cur_batch_size = tmp_pos.shape[0]
                if cur_batch_size == 0:
                    break

                tmp_input_params = np.concatenate(
                    [
                        np.zeros((cur_batch_size,3),np.float32),
                        tmp_axay,
                        tmp_pd,
                        tmp_ps
                    ],
                    axis=1
                )#(cur_batch_size,7/11)
                
                ##################################
                ###STEP 1 convert input to torch and render them
                #################################
                tmp_pos_tc = torch.from_numpy(tmp_pos).to(self.test_device)
                tmp_input_params_tc = torch.from_numpy(tmp_input_params).to(self.test_device)
                tmp_normal_tc = torch.from_numpy(tmp_normal).to(self.test_device)
                tmp_tangent_tc = torch.from_numpy(tmp_tangent).to(self.test_device)
                tmp_binormal_tc = torch.from_numpy(tmp_binormal).to(self.test_device)

                tmp_global_frame = [tmp_normal_tc,tmp_tangent_tc,tmp_binormal_tc]

                sampled_rotate_angles = torch.from_numpy(self.sampled_rotate_angles_np).repeat(cur_batch_size,1).to(self.test_device)
                sampled_rotate_angles = sampled_rotate_angles[:,[which_view]]#(cur_batch,1)

                rendered_lumi,_  = torch_render.draw_rendering_net(
                    self.setup,
                    tmp_input_params_tc,
                    tmp_pos_tc,
                    sampled_rotate_angles,
                    "test_view_{}".format(which_view),
                    global_custom_frame=tmp_global_frame,
                    use_custom_frame="ntb"
                )#(cur_batch_size,lightnum,1/3)
                rendered_lumi = rendered_lumi*self.RENDER_SCALAR#(cur_batch_size,lightnum,1)
                if not self.test_in_grey:
                    rendered_lumi = rendered_lumi.permute(0,2,1).reshape(cur_batch_size*3,self.setup.get_light_num(),1)#(cur_batch_size*3,lightnum,1)
                    sampled_rotate_angles = torch.unsqueeze(sampled_rotate_angles,dim=1).repeat(1,3,1).reshape(cur_batch_size*3,1)#(cur_batch_size*3,1)
                ##################################
                ###STEP 2 transefer lumi to dift codes
                #################################
                measurements = dift_trainer.linear_projection(rendered_lumi)#(cur_batch_size/cur_batch_size*3,measurement_len,1)
                cossin = torch.cat(
                        [
                            torch.sin(sampled_rotate_angles),
                            torch.cos(sampled_rotate_angles)
                        ],dim=1
                    )
                
                # view_mat_model = torch_render.rotation_axis(-sampled_rotate_angles,self.setup.get_rot_axis_torch(self.test_device))#[2*batch,4,4]
                # view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]
                # view_mat_model_t = view_mat_model_t.reshape(cur_batch_size,16) if self.test_in_grey else view_mat_model_t.reshape(cur_batch_size*3,16)
                # view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
                # view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[2*batch,4,4]
                # view_mat_for_normal_t = view_mat_for_normal_t.reshape(cur_batch_size,16) if self.test_in_grey else view_mat_for_normal_t.reshape(cur_batch_size*3,16)

                # albedo_nn_diff,albedo_nn_spec = dift_trainer.albedo_net(measurements_for_albedo)
                # if self.test_in_grey:
                #     albedo_nn_diff = tmp_input_params_tc[:,[5]]
                #     albedo_nn_spec = tmp_input_params_tc[:,[6]]
                # else:
                #     albedo_nn_diff = tmp_input_params_tc[:,5:8].reshape(cur_batch_size*3,1)
                #     albedo_nn_spec = tmp_input_params_tc[:,8:11].reshape(cur_batch_size*3,1)

                dift_codes_full,origin_code_map = dift_trainer.dift_net(measurements,cossin,True)#(batch,diftcodelen)/(batch*3,diftcodelen)
                
                if self.check_type == "a":
                    dift_codes = dift_codes_full
                else:
                    dift_codes = origin_code_map[self.check_type]
    
                dift_codes = dift_codes.cpu().numpy()
                dift_codes.astype(np.float32).tofile(pf_save)

                ptr+=cur_batch_size
            pf_save.close()

            #gen fake idex file
            with open(cur_save_root+"cam00_index_nocc.bin","wb") as pf:
                pf.write(struct.pack("i",self.valid_pixels_num))
                self.fake_index.tofile(pf)


        dift_trainer.to(dift_trainer.training_device)
        ##############################################################################
        ## compact dift codes for visualization and log them to tensorboard
        ##############################################################################
        the_process = Popen(
            [
                "python",
                "val_files/inference_dift/compact_dift_codes.py",
                self.log_dir,
                "{}".format(self.rotate_num),
                "{}".format((self.dift_code_len) if self.test_in_grey else (self.dift_code_len*3)),
                "3",
                "--test_on_the_fly",
                "--imgheight",
                "720",
                "--imgwidth",
                "1024"
            ]
        )
        exitcode = the_process.wait()
        if exitcode != 0:
            print("[DIFT QUALITY CHECKER] error occured in compact dift codes!")
            return
        
        ###logging
        imgs = []
        for which_view in range(self.rotate_num):
            tmp_img = cv2.imread(self.log_dir+"images_0/pd_predicted_{}_0.png".format(which_view))
            tmp_img = torch.from_numpy(tmp_img[:,:,::-1].copy())
            imgs.append(tmp_img)
        imgs = torch.stack(imgs,dim=0)
        imgs = imgs.permute(0,3,1,2)
        grid_image = torchvision.utils.make_grid(imgs, nrow=4, padding=5, pad_value=0)
        writer.add_image("{}".format(self.checker_name),grid_image, global_step=global_step, dataformats='CHW')

        ##############################################################################
        ## clean up
        ##############################################################################
        shutil.rmtree(self.log_dir)


if __name__ == "__main__":
    from torch_render import Setup_Config
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
    setup_input = Setup_Config(standard_rendering_parameters)

    fake_train_configs = {
        "setup_input":setup_input,
        "sample_view_num":24,
        "RENDER_SCALAR":5*1e3/math.pi
    }
    checker_undertest = DIFT_QUALITY_CHECKER(
        fake_train_configs,
        "./",
        "F:/CVPR21_freshmeat/SIGA20_data/sphere/undistort_feature_1pass_bak/metadata/",
        "test_checker",
        torch.device("cuda:0"),
        axay=(0.01,0.02),
        diff_albedo=0.3,
        spec_albedo=5.0
    )
