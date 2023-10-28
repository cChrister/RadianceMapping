from .net import MLP, UNet
from .mpn import MPN
import torch
import torch.nn as nn
from torchvision import transforms as T

class Renderer(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
        self.unet = UNet(args).to(args.device)
        self.mpn =  MPN(U=2, udim='pp', in_dim=args.dim * args.points_per_pixel).to(args.device)
        self.dim = args.dim

        # self.use_crop = False
        self.use_crop = True

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')
        
        self.xyznear = args.xyznear # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel= args.points_per_pixel

    def forward(self, zbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
            fea_map: the first three dimensions of the feature map of radiance mapping
        """

        radiance_map = []
        for i in range(zbufs.shape[-1]):
            zbuf = zbufs[..., i].unsqueeze(-1) # [H, W, 1]
            
            H, W, _ = zbuf.shape
            o = ray[..., :3] # [H, W, 1]
            dirs = ray[...,3:6] # [H, W, 3]
            cos = ray[...,-1:] # [H, W, 1]

            if isTrain:
                pix_mask = zbuf > 0.2 # [H, W, 1]
            else:
                pix_mask = zbuf > 0

            o = o.unsqueeze(-2).expand(H, W, 1, 3)[pix_mask] # occ_point 3
            dirs = dirs.unsqueeze(-2).expand(H, W, 1, 3)[pix_mask]  # occ_point 3
            cos = cos.unsqueeze(-2).expand(H, W, 1, 1)[pix_mask]  # occ_point 1
            zbuf = zbuf.unsqueeze(-1)[pix_mask]  # occ_point 1

            if self.xyznear:
                xyz_near = o + dirs * zbuf / cos # occ_point 3
            else:
                xyz_near = xyz_o[zbuf.squeeze(-1).long()]

            feature = self.mlp(xyz_near, dirs) # occ_point 3
            feature_map = torch.zeros([H, W, 1, self.dim], device=zbuf.device)
            feature_map[pix_mask] = feature # [400, 400, 1, self.dim]
            radiance_map.append(feature_map.permute(2, 3, 0, 1))


        rdmp = torch.cat(radiance_map, dim=1) # [1, self.dim * self.points_per_pixel, H, W]

        pred_mask = self.mpn(rdmp) # [1, self.dim * self.points_per_pixel, H, W]
        # fuse_rdmp = rdmp.mul(pred_mask).sum(dim=1)
        fuse_rdmp = rdmp.mul(pred_mask) # [1, self.dim * self.points_per_pixel, H, W]

        feature_map_view = fuse_rdmp.clone().squeeze(0)[:3, :, :]
        feature_map_view = torch.sigmoid(feature_map_view.permute(1, 2, 0))
        ret = self.unet(fuse_rdmp) # [1, 3, H, W]
        img = ret.squeeze(0).permute(1, 2, 0) # [H, W, 3]

        return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}

        # Unet
        # sigma H, W, 1, 1
        # color H, W, 1, 8
        # gt h w 3
        # pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)# h w 1 1
        # feature_map_view = torch.sigmoid(feature_map.clone().squeeze(-2)[...,:3])
        # feature_map = self.unet(feature_map.permute(2,3,0,1)) # [1, 3, 400, 400]
# 
        # if self.mask:
            # feature_map = feature_map * pix_mask + (1 - pix_mask) # 1 3 h w
        # img = feature_map.squeeze(0).permute(1,2,0)
# 
# 
        # return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}