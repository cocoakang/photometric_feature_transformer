import torch

def rotate_rowwise(matrix, shifts):
    """"
    requested rotate function - assumes matrix shape is mxnxd and shifts shape is m
    shift to left 
    example:
    input=tensor([[0.7606, 0.8259, 0.7414, 0.6489, 0.6051],
        [0.6412, 0.4258, 0.1738, 0.6976, 0.3856],
        [0.5356, 0.3621, 0.9968, 0.6937, 0.7057],
        [0.4632, 0.4416, 0.8622, 0.4147, 0.6623],
        [0.7161, 0.5564, 0.6787, 0.2689, 0.7222],
        [0.7358, 0.2367, 0.5559, 0.1869, 0.9007],
        [0.6862, 0.7950, 0.1362, 0.3240, 0.8282],
        [0.1680, 0.8386, 0.5726, 0.2541, 0.7545],
        [0.5015, 0.7893, 0.4242, 0.5765, 0.7414],
        [0.5714, 0.1799, 0.9927, 0.2483, 0.3669]])
    shifts = [1,3,2,2,4,4,4,1,1,2]
    result=tensor([[0.8259, 0.7414, 0.6489, 0.6051, 0.7606],
        [0.6976, 0.3856, 0.6412, 0.4258, 0.1738],
        [0.9968, 0.6937, 0.7057, 0.5356, 0.3621],
        [0.8622, 0.4147, 0.6623, 0.4632, 0.4416],
        [0.7222, 0.7161, 0.5564, 0.6787, 0.2689],
        [0.9007, 0.7358, 0.2367, 0.5559, 0.1869],
        [0.8282, 0.6862, 0.7950, 0.1362, 0.3240],
        [0.8386, 0.5726, 0.2541, 0.7545, 0.1680],
        [0.7893, 0.4242, 0.5765, 0.7414, 0.5015],
        [0.9927, 0.2483, 0.3669, 0.5714, 0.1799]])

    """
    
    # shifts = -1*shifts
    # get shape of the input matrix
    shape = matrix.size()
    device=matrix.device

    # compute and stack the meshgrid to get the index matrix of shape (2,m,n)
    ind = torch.stack(torch.meshgrid(torch.arange(shape[0],device=device), torch.arange(shape[1],device=device))).permute(1,2,0)
    # reshape it to (m,n,2)

    # add the value from shifts to the corresponding row and devide modulo shape[1]
    # this will effectively introduce the desired shift, but at the level of indices
    shifted_ind = torch.fmod(ind[:,:,1] + shifts, shape[1])
    
    # convert the shifted indices to the right shape
    new_ind = torch.stack([ind[:,:,0], shifted_ind])
    new_ind = new_ind.permute(1,2,0).view(-1,2)

    # return the resliced tensor
    return torch.reshape(matrix[new_ind[:,0],new_ind[:,1]],shape)