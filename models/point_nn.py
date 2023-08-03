# Non-Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

from .model_utils import *



# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        # xyz coordinate/ x features
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() # [16,512]
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN 在xyz中找lc_xyz的knn
        # knn_idx = knn_point_open3d(self.k_neighbors, xyz, lc_xyz)
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx) # [16,512,90,3]
        knn_x = index_points(x, knn_idx)# [16,512,90,72]

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # knn_x [16,512,90,72] / knn_xyz [16,512,90,3]

        # Normalize x (features) and xyz (coordinates)
        # 我有点好奇这边为什么要做归一化特征
        # torch.std 计算标准差，你这归一化做的？？
        # 用标准差来normalize 我看不懂
        mean_x = lc_x.unsqueeze(dim=-2)# [16,512,1,72]
        std_x = torch.std(knn_x - mean_x)# [0]=0.623

        mean_xyz = lc_xyz.unsqueeze(dim=-2)# [16,512,1,3]
        std_xyz = torch.std(knn_xyz - mean_xyz)# [0]=0.1368

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)# [16,512,90,72]
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)# [16,512,90,3]

        # Feature Expansion 即fcj
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        # [16,512,90,144]

        # Geometry Extraction
        # knn_x相当于论文中的fj，knn_xyz相当于论文中的PosE(Pj)
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x) # [16,144,512,90]

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())# 高斯误差函数

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        # alpha=1000, beta=100, in_dim=3, out_dim=72
        # torch.div 张量和标量做逐元素除法

    def forward(self, xyz):
        B, _, N = xyz.shape  # xyz shape [16,3,1024]
        feat_dim = self.out_dim // (self.in_dim * 2) # [12] sin和cos值共12个
        feat_range = torch.arange(feat_dim).float().cuda() # [12]  
        # 这里的alpha和beta和论文中是相反的
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)# [12]
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)# [16,3,1024,12]

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)# [16,3,1024,24]
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)# [16,72,1024]
        # 这里是怎么拼接和reshape的

        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim # 144
        self.alpha, self.beta = alpha, beta
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)
        # [16,144,512,90]

        # Weigh 论文中那一长串公式  element-wise multiplicatio
        # knn_x [16,144,512,90]
        # position_embed即PosE [16,144,512,90]
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        # 为什么进行knn之前要进行FPS
        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):
        # Raw-point Embedding
        x = self.raw_point_embed(x)# [16,72,1024]

        # Multi-stage Hierarchy
        for i in range(self.num_stages):# num_stages=4
            # FPS, kNN, xyz中点的数量会对半减少
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)
        """
        i=0, x[16,144,512], knn_x_w=[16,144,512,90]
        i=1, x[16,288,256], knn_x_w=[]
        i=2, x[16,576,128]
        i=3, x[16,1152,64]
        """
        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1) # [16,1152]
        return x


# Non-Parametric Network
class Point_NN(nn.Module):
    def __init__(self, input_points=1024, num_stages=4, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)


    def forward(self, x):
        # xyz: point coordinates
        # x: point features 什么逆天变量名
        xyz = x.permute(0, 2, 1) # [16,1024,3]

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)
        return x
