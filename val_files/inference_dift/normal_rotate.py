import numpy as np
import cv2
import os
import math

def rotation_axis_numpy(t,v,isRightHand=True):
    '''
    t = [batch,1]#rotate rad??
    v = [batch,3]#rotate axis(global) 
    return = [batch,4,4]#rotate matrix
    '''
    if isRightHand:
        theta = t
    else:
        print("[RENDERER]Error rotate system doesn't support left hand logic!")
        exit()
    assert t.shape[0] == v.shape[0]
    fitting_batch_size = t.shape[0]
    
    c = np.cos(theta)
    s = np.sin(theta)

    v_x = v[:,[0]]
    v_y = v[:,[1]]
    v_z = v[:,[2]]

    m_11 = c + (1-c)*v_x*v_x
    m_12 = (1 - c)*v_x*v_y - s*v_z
    m_13 = (1 - c)*v_x*v_z + s*v_y

    m_21 = (1 - c)*v_x*v_y + s*v_z
    m_22 = c + (1-c)*v_y*v_y
    m_23 = (1 - c)*v_y*v_z - s*v_x

    m_31 = (1 - c)*v_z*v_x - s*v_y
    m_32 = (1 - c)*v_z*v_y + s*v_x
    m_33 = c + (1-c)*v_z*v_z

    tmp_zeros = np.zeros([fitting_batch_size,1])
    tmp_ones = np.ones([fitting_batch_size,1])

    res = np.concatenate([
        m_11,m_12,m_13,tmp_zeros,
        m_21,m_22,m_23,tmp_zeros,
        m_31,m_32,m_33,tmp_zeros,
        tmp_zeros,tmp_zeros,tmp_zeros,tmp_ones
    ],axis=-1)

    res = np.reshape(res,[fitting_batch_size,4,4])
    return res

if __name__ == "__main__":
    roo = "/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/buddhaPNG/"
    root = roo+"infered_normals/"
    save_root = roo+"normal_exrs_globalT/"

    os.makedirs(save_root,exist_ok=True)

    angles = np.linspace(0.0,math.pi*2.0,num=20,endpoint=False)

    for which_view in range(20):
        data = cv2.imread(root+"normal_infered_{}.png".format(which_view))[:,:,::-1]
        data = data.astype(np.float32)/255.0
        data_origin = data.copy()
        data = data*2-1.0

        data = np.reshape(data,[-1,3])
        data = data[:,::-1]
        # data = np.repeat(np.array([[1.0,0.0,0.0]],np.float32),data.shape[0],axis=0)

        v = np.repeat(np.array([[0,1,0]],np.float32),data.shape[0],axis=0)
        t = np.ones([data.shape[0],1],np.float32)*angles[which_view]
        rotate_matrix = rotation_axis_numpy(t,v)[0]#[4,4]

        data_homo = np.concatenate([data,np.ones([data.shape[0],1],np.float32)],axis=-1)

        data_homo = np.matmul(data_homo,rotate_matrix)
        
        data = data_homo[:,:3]

        data = data * 0.5 + 0.5

        data = np.reshape(data,[512,612,3])

        mask = np.where(data_origin>0.0,np.ones_like(data_origin),np.zeros_like(data_origin))

        data_new = np.zeros_like(data)
        np.copyto(data_new,data,where=mask>0.0)

        data_new = data_new[:,:,::-1]
        print(data.dtype)
        # cv2.imwrite(save_root+"{}.exr".format(which_view),data_new.astype(np.float32))
        cv2.imwrite(save_root+"pd_predicted_{}.png".format(which_view),data_new.astype(np.float32)*255.0)