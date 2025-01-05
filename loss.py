import torch.nn.functional as F
import torch
from torch import einsum
import numpy as np
import torch.nn as nn

def pairwise_euclidean_distance_2d(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j, label):
        diff = torch.pow(emb_i-emb_j,2)
        # diff = torch.abs(emb_i-emb_j)
        diff = torch.mean(diff,dim=1)
        loss = diff.mean()
        return loss

class DistillationLoss(nn.Module):
    def __init__(self, temperature=5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy1 = nn.CrossEntropyLoss()
        self.cross_entropy2 = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, student_outputs, teacher_outputs, labels, student_fts=None, teacher_fts=None,ce_weight=1.0, cl_weight=1.0,kernel_mul=2.0):
        # 计算交叉熵损失
        ce_loss = self.cross_entropy1(teacher_outputs, labels) # self.cross_entropy(student_outputs, labels) + 

        # 计算软标签
        soft_labels = nn.functional.softmax(teacher_outputs / self.temperature, dim=1)

        ce_loss1 = self.cross_entropy2(student_outputs, soft_labels)


        if student_fts is not None:
            mkmmd_loss1 = mmd_rbf(student_fts,teacher_fts,kernel_mul = kernel_mul)
            distillation_loss = ce_loss + ce_weight * ce_loss1 + cl_weight * mkmmd_loss1
        else:
            # 综合交叉熵损失和 KL 散度损失作为最终的蒸馏损失
            distillation_loss =  ce_weight * ce_loss1

        return distillation_loss
    

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵,即上文中的K
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算
