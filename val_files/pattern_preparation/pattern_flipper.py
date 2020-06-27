import numpy as np
import sys

lumitexel_size = 24576
channel_num = 3

if __name__ == "__main__":
    modelPath = sys.argv[1]
    sample_view_num = int(sys.argv[2])
    all_pos = int(sys.argv[3]) == 1

    for which_view in range(sample_view_num):
        cur_root = modelPath+"{}/".format(which_view)
        W = np.fromfile(cur_root+"W.bin",np.float32).reshape([-1,lumitexel_size,3])#(pattern_num,light_num,3)
        patternNum = W.shape[0]
        print("origin pattern num:",patternNum)
        
        if all_pos:
            W_flipped = W
        else:
            W_flipped = np.zeros([patternNum*2,lumitexel_size,channel_num])
            W_flipped = np.stack(
                [
                    np.maximum(0.0,W),
                    np.minimum(0.0,W)*-1.0
                ],axis=1
            )
            W_flipped = np.reshape(W_flipped,[patternNum,2,lumitexel_size,channel_num])#[pattern_num*2(pos neg), lumisize,3]
        W_flipped.astype(np.float32).tofile(cur_root+"W_flipped.bin")
