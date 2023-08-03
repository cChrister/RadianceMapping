import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import time

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # src = src.to('cpu')
    # dst = dst.to('cpu')
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # dist = dist.to('cuda:0')
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    # idx = idx.to('cpu')
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # batch_indices = torch.arange(B, dtype=torch.long).to('cpu').view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    new_points = new_points.to(device)
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def knn_point_open3d(nsample, xyz, new_xyz):
    device=xyz.device
    pcd_array = xyz[0]
    queries_array = new_xyz[0]

    points = pcd_array.to('cpu')
    queries = queries_array.to('cpu')
    num_queries = queries_array.shape[0]
    
    # we use open3d to knn
    # start = time.time()
    nsearch = ml3d.layers.KNNSearch()
    ans = nsearch(points, queries, nsample)
    # end = time.time()

    neighbors_index = ans.neighbors_index.reshape(num_queries, nsample)
    neighbors_index = neighbors_index.unsqueeze(0).type(torch.long)
    neighbors_index = neighbors_index.to(device)
    return neighbors_index
    