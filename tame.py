from mine_pro import Mine_Pro
import torch
from torch.utils.tensorboard import SummaryWriter 
import torchvision
import torch.optim as optim
from DIFT_TRAIN_NET import DIFT_TRAIN_NET
import argparse
import time
import sys
import random
import numpy as np
import os
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import Setup_Config
import queue


MAX_ITR = 5000000
VALIDATE_ITR = 5
CHECK_QUALITY_ITR=1000
SAVE_MODEL_ITR=10000
LOG_MODEL_ITR=100000

def log_loss(writer,loss_terms,global_step,is_training):
    train_val_postfix = "_train" if is_training else "_val"

    for a_key in loss_terms:
        if "loss_e1_train_tamer" in a_key and is_training:
            writer.add_scalar('loss_e1_train_tamer', loss_terms[a_key], global_step)
        elif "loss_e1_train_tamer" in a_key and not is_training:
            writer.add_scalar('loss_e1_val_tamer', loss_terms[a_key], global_step)
        else:
            writer.add_scalar('{}'.format(a_key)+train_val_postfix, loss_terms[a_key], global_step)

def log_quality(writer,quality_terms,global_step):
    term_key = "multiview_lumi_img"
    batch_size = quality_terms[term_key].shape[0]
    # writer.add_images(term_key, quality_terms[term_key], global_step=global_step, dataformats='NHWC')
    for which_sample in range(batch_size):
        writer.add_image("{}/{}".format(term_key,which_sample), quality_terms[term_key][which_sample],  global_step=global_step, dataformats='CHW')

    term_key = "lighting_pattern_rgb"
    m_len = quality_terms[term_key].shape[0]
    for which_m in range(m_len):
        grid_image = torchvision.utils.make_grid((torch.from_numpy(quality_terms[term_key][which_m])).permute(0,3,1,2), nrow=4, padding=5, pad_value=0)
        writer.add_image("{}/{}".format(term_key,which_m),grid_image, global_step=global_step, dataformats='CHW')
    
    term_key = "distance_matrix"
    writer.add_image("{}".format(term_key),quality_terms[term_key], global_step=global_step, dataformats='CHW')

if __name__ == "__main__":
    torch.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--training_gpu",type=int,default=1)
    parser.add_argument("--rendering_gpu",type=int,default=1)
    parser.add_argument("--sample_view_num",type=int,default=24)
    parser.add_argument("--measurement_num",type=int,default=4)
    parser.add_argument("--m_noise_rate",type=float,default=0.01)
    parser.add_argument("--dift_code_len",type=int,default=128)
    parser.add_argument("--log_file_name",type=str,default="")
    parser.add_argument("--pretrained_model_pan",type=str,default="")

    args = parser.parse_args()
    
    ##########################################
    ### parser training configuration
    ##########################################

    ##about rendering devices
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
    setup_input = Setup_Config(standard_rendering_parameters)

    ##build train_configs
    train_configs = {}
    train_configs["rendering_device"] = torch.device("cuda:{}".format(args.rendering_gpu))
    train_configs["training_device"] = torch.device("cuda:{}".format(args.training_gpu))
    train_configs["sample_view_num"] = args.sample_view_num
    train_configs["lumitexel_downsample_rate"] = 1
    train_configs["lumitexel_length"] = 24576 // train_configs["lumitexel_downsample_rate"] // train_configs["lumitexel_downsample_rate"]
    train_configs["measurements_length"] = args.measurement_num
    train_configs["noise_stddev"] = args.m_noise_rate
    train_configs["setup_input"] = setup_input
    train_configs["dift_code_len"] = args.dift_code_len

    lambdas = {}
    lambdas["diff"] =1.0
    lambdas["weight"] = 1e-3
    train_configs["lambdas"] = lambdas

    train_configs["data_root"] = args.data_root
    train_configs["batch_size"] = 25
    train_configs["pre_load_buffer_size"] = 500000

    
    ##########################################
    ### data loader
    ##########################################
    # train_Semaphore = Semaphore(100)
    train_queue = queue.Queue(25)
    # val_Semaphore = Semaphore(50)
    val_queue = queue.Queue(10)
    train_mine = Mine_Pro(train_configs,"train",train_queue,None)
    train_mine.start()
    val_mine = Mine_Pro(train_configs,"val",val_queue,None)
    val_mine.start()
    ##########################################
    ### net and optimizer
    ##########################################
    training_net = DIFT_TRAIN_NET(train_configs)
    training_net.to(train_configs["training_device"])
    
    optimizer = optim.Adam(training_net.parameters(), lr=1e-4,weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.9)

    ##########################################
    ### define others
    ##########################################
    if args.log_file_name == "":
        writer = SummaryWriter(comment="dift_rgb")
        # os.makedirs("../log_no_where/",exist_ok=True)
        # os.system("rm -r ../log_no_where/*")
        # writer = SummaryWriter(log_dir="../log_no_where/")
    else:
        writer = SummaryWriter(args.log_file_name)
    log_dir = writer.get_logdir()
    log_dir_model = log_dir+"/models/"
    os.makedirs(log_dir_model,exist_ok=True)
    train_file_bak = log_dir+"/training_files/"
    os.makedirs(train_file_bak)
    os.system("cp *.py "+train_file_bak)
    os.system("cp *.sh "+train_file_bak)
    with open(log_dir_model+"model_params.txt","w") as pf:
        pf.write("{}".format(training_net))
        pf.write("-----------------")
        for parameter in training_net.parameters():
            pf.write("{}\n".format(parameter.shape))
    start_step = 0
    ##########################################
    ### load models
    ##########################################
    if args.pretrained_model_pan != "":
        print("loading trained model...")
        state = torch.load(args.pretrained_model_pan, map_location=torch.device('cpu'))
        training_net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state["lr_schedular"])
        start_step = state['epoch']
        training_net.to(train_configs["training_device"])
        print("done.")
        
    ##########################################
    ### training part
    ##########################################
    for global_step in range(start_step,MAX_ITR):
        if global_step % 100 == 0:
            print("global step:{}".format(global_step))
        ## 1 validate
        if global_step % VALIDATE_ITR == 0:
            # print("val queue size:",val_queue.qsize())
            val_data = val_queue.get()
            # val_Semaphore.release()
            # print("got val")
            with torch.no_grad():
                training_net.eval()
                _,loss_log_terms = training_net(val_data,global_step=global_step)
            log_loss(writer,loss_log_terms,global_step,False)
            
        ## 2 check quality
        if global_step % CHECK_QUALITY_ITR == 0:
            # print("val queue size:",val_queue.qsize())
            val_data = val_queue.get()
            # val_Semaphore.release()
            # print("got check")
            with torch.no_grad():
                training_net.eval()
                quality_terms = training_net(val_data,call_type="check_quality",global_step=global_step)
            log_quality(writer,quality_terms,global_step)
        ## 3 save model
        if global_step % SAVE_MODEL_ITR == 0 and global_step != 0:
            training_state = {
                'epoch': global_step,
                'state_dict': training_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedular':scheduler.state_dict()
            }
            torch.save(training_state, log_dir_model+"training_state.pkl")
            
        if global_step % LOG_MODEL_ITR == 0 and global_step != 0:
            torch.save(training_net.state_dict(), log_dir_model+"model_state_{}.pkl".format(global_step))

        ## 4 training
        training_net.train()
        optimizer.zero_grad()
        # print("train queue size:",train_queue.qsize())
        # start = time.time()
        # end_time = [start]
        train_data = train_queue.get()
        # end_time.append(time.time())
        # train_Semaphore.release()
        # print("got train")
        total_loss,loss_log_terms = training_net(train_data,global_step=global_step)
        # end_time.append(time.time())
        total_loss.backward()
        # end_time.append(time.time())
        optimizer.step()
        # end_time.append(time.time())
        # total_time = end_time[-1] - start
        # time_elspase = [end_time[i]-end_time[i-1] for i in range(1,len(end_time))]
        # time_ratio = [a/total_time for a in time_elspase]
        # for i in range(len(time_elspase)):
        #     print("{:0.3} {:0.3} |".format(time_elspase[i],time_ratio[i]),end="")
        # print("\n")
        # if global_step > 1000000:
        #     scheduler.step()
        loss_log_terms["lr"] = optimizer.param_groups[0]['lr']
        log_loss(writer,loss_log_terms,global_step,True)
        # if global_step == 14526:
        #     break
        # break
    writer.close()
