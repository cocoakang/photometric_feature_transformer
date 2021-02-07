from mine_pro import Mine_Pro
import torch
from torch.utils.tensorboard import SummaryWriter 
import torchvision
import torch.optim as optim
from DIFT_TRAIN_NET import DIFT_TRAIN_NET
from DIFT_QUALITY_CHECKER import DIFT_QUALITY_CHECKER
import argparse
import time
import sys
import random
import numpy as np
import os
TORCH_RENDER_PATH="../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import Setup_Config_Freeform
from multiprocessing import Queue
import math
import re

MAX_ITR = 300000
VALIDATE_ITR = 5
CHECK_QUALITY_ITR=5000
SAVE_MODEL_ITR=10000
LOG_MODEL_ITR=30000

def log_loss(writer,loss_terms,global_step,is_training,post_fix=""):
    train_val_postfix = "_train" if is_training else "_val"
    if not is_training:
        train_val_postfix = "_"+post_fix+train_val_postfix

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

def parse_vh_config(pretrained_model_pan_h,pretrained_model_pan_v):
    x = re.search("learn_l2_ml", pretrained_model_pan_h)
    start_idx = x.start()+len("learn_l2_ml")
    m_m_len = int(pretrained_model_pan_h[start_idx:start_idx+1])
    x = re.search("_dlna", pretrained_model_pan_h)
    start_idx = x.start()+len("_dlna")
    m_d_len = int(pretrained_model_pan_h[start_idx:start_idx+1])

    x = re.search("_mg", pretrained_model_pan_v)
    start_idx = x.start()+len("_mg")
    g_m_len = int(pretrained_model_pan_v[start_idx:start_idx+1])
    x = re.search("_dg", pretrained_model_pan_v)
    start_idx = x.start()+len("_dg")
    g_d_len = int(pretrained_model_pan_v[start_idx:start_idx+1])

    return m_m_len,m_d_len,g_m_len,g_d_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--training_gpu",type=int,default=1)
    parser.add_argument("--rendering_gpu",type=int,default=1)
    parser.add_argument("--checker_gpu",type=int,default=1)
    parser.add_argument("--log_file_name",type=str,default="")
    parser.add_argument("--pretrained_model_pan",type=str,default="")
    parser.add_argument("--pretrained_model_pan_h",type=str,default="")
    parser.add_argument("--pretrained_model_pan_v",type=str,default="")
    parser.add_argument("--start_seed",type=int,default=84057)
    parser.add_argument("--torch_manual_seed",type=int,default=1827397)
    parser.add_argument("--torch_cuda_manual_seed_all",type=int,default=1827397)
    parser.add_argument("--train_mine_seed",type=int,default=51721)
    parser.add_argument("--val_mine_seed",type=int,default=992831)
    parser.add_argument("--search_model",action="store_true")
    parser.add_argument("--m_len",type=int,default=3)
    parser.add_argument("--code_len",type=int,default=5)
    parser.add_argument("--id",type=int,default=-1)
    parser.add_argument("--search_which",default="geometry",choices=["material","geometry"])

    args = parser.parse_args()

    start_seed = args.start_seed
    torch.manual_seed(args.torch_manual_seed)
    torch.cuda.manual_seed_all(args.torch_cuda_manual_seed_all)
    random.seed(start_seed)
    np.random.seed(start_seed)

    if args.search_model:
        MAX_ITR=100000
    
    ##########################################
    ### parser training configuration
    ##########################################

    ##about rendering devices
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/diligent_mv/"
    }
    setup_input = Setup_Config_Freeform(standard_rendering_parameters)
    setup_input.rot_axis = np.array([0.0,1.0,0.0],np.float32)# TODO read from calibration file
    setup_input2 = Setup_Config_Freeform(standard_rendering_parameters)
    setup_input2.rot_axis = np.array([-1.0,0.0,0.0],np.float32)# TODO read from calibration file

    ##build train_configs
    train_configs = {}
    train_configs["rendering_device"] = torch.device("cuda:{}".format(args.rendering_gpu))
    train_configs["training_device"] = torch.device("cuda:{}".format(args.training_gpu))
    train_configs["sample_view_num_whentest"] = 24
    train_configs["lumitexel_downsample_rate"] = 1
    train_configs["lumitexel_length"] = 24576 // train_configs["lumitexel_downsample_rate"] // train_configs["lumitexel_downsample_rate"]
    train_configs["noise_stddev"] = 0.01
    train_configs["setup_input"] = setup_input
    train_configs["setup_input2"] = setup_input2
    train_configs["training_mode"] = "pretrain" if (args.pretrained_model_pan_h == "" and args.pretrained_model_pan_v == "") else "finetune"

    train_configs["RENDER_SCALAR"] = 5*1e3/math.pi

    partition = {}#m_len
    dift_code_config = {}#dift_code_len,losslambda
    if args.search_model:
        print("not ready")
        exit()
        if args.search_which == "material":
            partition["local"] = args.m_len
            partition["global"] = 0
            dift_code_config["local_noalbedo"] = (args.code_len,1.0)
            dift_code_config["global"] = (0,10.0)
        elif args.search_which == "geometry":
            partition["local"] = 0
            partition["global"] = args.m_len
            dift_code_config["local_noalbedo"] = (0,1.0)
            dift_code_config["global"] = (args.code_len,10.0)
        else:
            print("unkown search type")
    else:
        if train_configs["training_mode"] == "pretrain":
            partition["local"] = 0
            partition["global"] = setup_input.get_light_num()
            dift_code_config["local_noalbedo"] = (0,1.0)
            dift_code_config["global"] = (5,10.0)
        elif train_configs["training_mode"] == "finetune":
            print("not ready")
            exit()
            m_m_len,m_d_len,g_m_len,g_d_len = parse_vh_config(args.pretrained_model_pan_h,args.pretrained_model_pan_v)
            partition["local"] = m_m_len
            partition["global"] = g_m_len
            dift_code_config["local_noalbedo"] = (m_d_len,1.0)
            dift_code_config["global"] = (g_d_len,10.0)
            dift_code_config["cat"] = (m_d_len+g_d_len,10.0)

    train_configs["measurements_length"] = setup_input.get_light_num()#sum([partition[a_key] for a_key in partition])
    train_configs["partition"] = partition
    train_configs["dift_code_config"] = dift_code_config
    if train_configs["training_mode"] == "pretrain":
        train_configs["dift_code_len"] = sum([dift_code_config[a_key][0] for a_key in dift_code_config if a_key != "cat"])
    else:
        print("not ready")
        exit()
        train_configs["dift_code_len"] = dift_code_config["cat"][0]

    lambdas = {}
    lambdas["E1"] = 1.0
    # lambdas["E2"] =1e-1
    lambdas["reg_loss"] = 1.0
    train_configs["lambdas"] = lambdas

    train_configs["global_data_loss"] = 1.0
    train_configs["local_data_loss"] = 1.0

    train_configs["data_root"] = args.data_root
    train_configs["batch_size"] = 25
    train_configs["pre_load_buffer_size"] = 500000

    ##########################################
    ### data loader
    ##########################################
    # train_Semaphore = Semaphore(100)
    # train_queue_local = Queue(10)
    train_queue_global = Queue(10)
    # val_Semaphore = Semaphore(50)
    val_queue = Queue(3)
    # val_queue_mine_v = Queue(3)
    # val_queue_mine_h = Queue(3)
    # val_queue_mine_a = Queue(3)
    tmp_noise_config_train_hard = {
        "position" : 2,
        "frame_normal_v" : 50.0,
        "frame_normal_h" : 50.0,
        "theta" : 0.1,
        "axay" : 0.1,
        "pd" : 0.1,
        "ps" : 0.1
    }
    train_mine_global = Mine_Pro(train_configs,"train",train_queue_global,None,args.train_mine_seed)
    train_mine_global.start()
    val_mine = Mine_Pro(train_configs,"val",val_queue,None,args.val_mine_seed,None)
    val_mine.start()
    
    ##########################################
    ### net and optimizer
    ##########################################
    training_net = DIFT_TRAIN_NET(train_configs)
    training_net.to(train_configs["training_device"])

    if train_configs["training_mode"] == "finetune":
        training_net.load_pretrained_models(args.pretrained_model_pan_h,args.pretrained_model_pan_v)

    lr = 1e-4# if train_configs["training_mode"] == "pretrain" else 1e-5
    optimizer = optim.Adam(training_net.parameters(), lr=lr,weight_decay=1e-6)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.9)

    ##########################################
    ### define others
    ##########################################
    if args.log_file_name == "":
        # writer = SummaryWriter(log_dir="runs/diligent_normal")
        os.makedirs("../log_no_where2/",exist_ok=True)
        os.system("rm -r ../log_no_where2/*")
        writer = SummaryWriter(log_dir="../log_no_where2/")
    else:
        writer = SummaryWriter(args.log_file_name)
    log_dir = writer.get_logdir()
    log_dir_model = log_dir+"/models/"
    os.makedirs(log_dir_model,exist_ok=True)
    train_file_bak = log_dir+"/training_files/"
    os.makedirs(train_file_bak)
    os.system("cp *.py "+train_file_bak)
    os.system("cp *.sh "+train_file_bak)
    os.system("cp *.txt "+train_file_bak)
    with open(log_dir_model+"model_params.txt","w") as pf:
        pf.write("{}".format(training_net))
        pf.write("-----------------")
        for parameter in training_net.parameters():
            pf.write("{}\n".format(parameter.shape))
    with open(log_dir_model+"seeds.txt","w") as pf:
        pf.write("{} {} {} {} {}\n".format(args.start_seed,args.torch_manual_seed,args.train_mine_seed,args.start_seed,args.val_mine_seed))
        pf.write("m_len:{} code_len:{}\n".format(args.m_len,args.code_len))
    ###quality checker
    quality_checkers = []

    # if dift_code_config["local_noalbedo"][0] > 0:
    #     checker_uniform_mirror_ball = DIFT_QUALITY_CHECKER(
    #         train_configs,
    #         log_dir,
    #         "../../training_data/feature_pattern_models/uniform_mirror_ball/metadata/",
    #         "uniform_mirror_ball_gh",
    #         torch.device("cuda:{}".format(args.checker_gpu)),
    #         axay=(0.05,0.05),
    #         diff_albedo=0.5,
    #         spec_albedo=3.0,
    #         batch_size=500,
    #         test_view_num=1,
    #         check_type="local_noalbedo"
    #     )
    #     quality_checkers.append(checker_uniform_mirror_ball)

    # if dift_code_config["global"][0] > 0:
    #     checker_uniform_mirror_ball = DIFT_QUALITY_CHECKER(
    #         train_configs,
    #         log_dir,
    #         "../../training_data/feature_pattern_models/uniform_mirror_ball/metadata/",
    #         "uniform_mirror_ball_gv",
    #         torch.device("cuda:{}".format(args.checker_gpu)),
    #         axay=(0.05,0.05),
    #         diff_albedo=0.5,
    #         spec_albedo=3.0,
    #         batch_size=500,
    #         test_view_num=1,
    #         check_type="global"
    #     )
    #     quality_checkers.append(checker_uniform_mirror_ball)

    checker_uniform_mirror_ball = DIFT_QUALITY_CHECKER(
        train_configs,
        log_dir,
        "../../training_data/feature_pattern_models/uniform_mirror_ball/metadata/",
        "uniform_mirror_ball_a",
        torch.device("cuda:{}".format(args.checker_gpu)),
        axay=(0.05,0.05),
        diff_albedo=0.5,
        spec_albedo=3.0,
        batch_size=500,
        test_view_num=1,
        check_type="a"
    )
    quality_checkers.append(checker_uniform_mirror_ball)

    # # checker_uniform_mirror_ball = DIFT_QUALITY_CHECKER(
    # #     train_configs,
    # #     log_dir,
    # #     "../../training_data/feature_pattern_models/uniform_mirror_ball/metadata/",
    # #     "uniform_mirror_ball_m",
    # #     torch.device("cuda:{}".format(args.checker_gpu)),
    # #     axay=(0.05,0.05),
    # #     diff_albedo=0.5,
    # #     spec_albedo=3.0,
    # #     batch_size=500,
    # #     test_view_num=1,
    # #     check_type="global"
    # # )
    # # quality_checkers.append(checker_uniform_mirror_ball)

    # # checker_textured_ball_1 = DIFT_QUALITY_CHECKER(
    # #     train_configs,
    # #     log_dir,
    # #     "../../training_data/feature_pattern_models/textured_ball_1/metadata/",
    # #     "textured_ball_1",
    # #     torch.device("cuda:{}".format(args.checker_gpu)),
    # #     batch_size=500,
    # #     test_view_num=1,
    # #     check_type="local_albedo"
    # # )
    # # quality_checkers.append(checker_textured_ball_1)

    # # checker_golden_pig = DIFT_QUALITY_CHECKER(
    # #     train_configs,
    # #     log_dir,
    # #     "../../training_data/feature_pattern_models/golden_pig/metadata/",
    # #     "golden_pig_a",
    # #     torch.device("cuda:{}".format(args.checker_gpu)),
    # #     batch_size=500,
    # #     test_view_num=1,
    # #     test_in_grey=False,
    # #     check_type="a"
    # # )
    # # quality_checkers.append(checker_golden_pig)


    start_step = 0
    ##########################################
    ### load models
    ##########################################
    if args.pretrained_model_pan != "":
        print("loading trained model...")
        state = torch.load(args.pretrained_model_pan, map_location=torch.device('cpu'))
        try:
            training_net.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state["lr_schedular"])
            start_step = state['epoch']
        except Exception as identifier:
            print("cannot found key, try to load parameter directly")
            # training_net.load_state_dict(state)
            model_dict = training_net.state_dict()
            pretrained_dict = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            training_net.load_state_dict(model_dict)

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
            # val_itr = global_step // VALIDATE_ITR
            val_data = val_queue.get()
            post_fix = "general"
            # if val_itr % 2 == 0:
            #     val_data = val_queue.get()
            #     post_fix = "general"
            # elif val_itr % 2 == 1:
            #     val_data = val_queue_mine_a.get()
            #     post_fix = "LDCa"
            # elif val_itr % 4 == 2:
            #     val_data = val_queue_mine_v.get()
            #     post_fix = "LDCv"
            # elif val_itr % 4 == 3:
            #     val_data = val_queue_mine_h.get()
            #     post_fix = "LDCh"
            # val_Semaphore.release()
            # print("got val")
            with torch.no_grad():
                training_net.eval()
                _,loss_log_terms = training_net(val_data,global_step=global_step,call_type="val")
            log_loss(writer,loss_log_terms,global_step,False,post_fix=post_fix)

        ## 2 check quality
        if global_step % CHECK_QUALITY_ITR == 0 or global_step == 1000:
            # print("val queue size:",val_queue.qsize())
            # val_data = val_queue.get()
            # val_Semaphore.release()
            # print("got check")
            # with torch.no_grad():
            #     training_net.eval()
            #     quality_terms = training_net(val_data,call_type="check_quality",global_step=global_step)
            # log_quality(writer,quality_terms,global_step)

            #check with real(real fake) data
            with torch.no_grad():
                for a_checker in quality_checkers:
                    print("========")
                    training_net.eval()
                    a_checker.check_quality(training_net,writer,global_step)

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
        train_data_global = train_queue_global.get()
        # end_time.append(time.time())
        # train_Semaphore.release()
        # print("got train")
        tmp_total_loss,tmp_loss_log_terms = training_net(train_data_global,global_step=global_step)
        total_loss = tmp_total_loss
        loss_log_terms = tmp_loss_log_terms
        
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
    print("o wa ru yo~")