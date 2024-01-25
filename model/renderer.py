from .net import MLP, UNet, UNet_color, UNet_super,AFNet 
from .mpn import MPN, MPN_tiny
import torch
import torch.nn as nn
from torchvision import transforms as T
# profile
from torchsummary import summary
from utils import config_parser
import time


class Renderer(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

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

        H, W, _ = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 1]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:]  # [H, W, 1]

        _o = o.unsqueeze(-2).expand(H, W, 1, 3)
        _dirs = dirs.unsqueeze(-2).expand(H, W, 1, 3)
        _cos = cos.unsqueeze(-2).expand(H, W, 1, 3)

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11:].permute(1, 2, 0)
            else:
                zbufs_crop = cat_img[0, 10:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            _o = ray[:H, :W, :3].unsqueeze(-2)
            _dirs = dirs_crop.clone().unsqueeze(-2).expand(H, W, 1, 3)
            _cos = cos_crop.clone().unsqueeze(-2).expand(H, W, 1, 3)

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad
            torch.cuda.empty_cache()

        
        radiance_map = []
        for i in range(zbufs.shape[-1]):
            zbuf = zbufs[..., i].unsqueeze(-1)  # [H, W, 1]

            if isTrain:
                pix_mask = zbuf > 0.2  # [H, W, 1]
            else:
                pix_mask = zbuf > 0

            o = _o[pix_mask]  # occ_point 3
            dirs = _dirs[pix_mask]  # occ_point 3
            cos = _cos[pix_mask]  # occ_point 1
            zbuf = zbuf.unsqueeze(-1)[pix_mask]  # occ_point 1

            if self.xyznear:
                xyz_near = o + dirs * zbuf / cos  # occ_point 3
            else:
                xyz_near = xyz_o[zbuf.squeeze(-1).long()]

            feature = self.mlp(xyz_near, dirs)  # occ_point 3

            feature_map = torch.zeros([H, W, 1, self.dim], device=zbuf.device)
            feature_map[pix_mask] = feature  # [400, 400, 1, self.dim]
            radiance_map.append(feature_map.permute(2, 3, 0, 1))

        # [1, self.dim * self.points_per_pixel, H, W]
        rdmp = torch.cat(radiance_map, dim=1)

        ################################### print(f"rdmp {t_cnt}") # 0.2s


        del radiance_map, _o, _dirs, _cos
        torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(rdmp)

        # confidence = pred_mask.clone().detach()
        # confidence = confidence.mean(dim=1)

        ###################################     try to add regulize term #################################
        # for i in range(self.points_per_pixel):
            # pred_mask[:, i*self.dim:(i+1)*self.dim, :, :] #[1, dim, H, W]

        # [1, self.dim * self.points_per_pixel, H, W]
        
        fuse_rdmp = rdmp.mul(pred_mask)
        # fuse_rdmp = rdmp.mul(pred_mask).sum(dim=1).unsqueeze(1) # [1, 1, H, W]
        # fuse_rdmp = fuse_rdmp.expand(1, self.dim, H, W)

        ################################## print(f"mpn {t_cnt}") # 0.01s


        # feature_map_view = fuse_rdmp.clone().squeeze(0)[:3, :, :]
        # feature_map_view = torch.sigmoid(feature_map_view.permute(1, 2, 0))
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]
        ################################## print(f"unet {t_cnt}") # 0.01s

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        # return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

class Renderer_color(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_color, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=3 * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=3 *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet_color(args).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, color, gt, mask_gt, isTrain):
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

        H, W, C = color.shape  # [H, W, points_per_pixel]

        if self.use_crop and isTrain:
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            color_pad = self.pad_w(color.permute(2, 0, 1).unsqueeze(0))
            
            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([color_pad, gt_pad, mask_gt], dim=1)
            else:
                cat_img = torch.cat([color_pad, gt_pad], dim=1)

            cat_img = self.randomcrop(cat_img)

            color = cat_img[0, :C].permute(1, 2, 0) # [h, w, 3 * points_per_pixel]
            gt = cat_img[0, C:C+3].permute(1, 2, 0) # [h, w, 3]

            
            if mask_gt is not None:
                mask_gt = cat_img[0, C+3:].permute(1, 2, 0)

        # [1, 3 * burst, H, W]
        rdmp = color.permute(2, 0, 1).unsqueeze(0)

        # [1, 3 * self.points_per_pixel, H, W]
        pred_mask = self.mpn(rdmp)

        # initial alpha-blending
        # alpha = torch.cumprod((1 - pred_mask),dim=1)[:,:-1,:,:]
        # ones = (torch.ones_like(pred_mask).to(pred_mask.device)[:,0,:,:]).unsqueeze(0)
        # alpha = torch.cat((ones,alpha),1)
        # fuse_rdmp = rdmp.mul(pred_mask).mul(alpha)

        # [1, 3 * self.points_per_pixel, H, W]
        fuse_rdmp = rdmp.mul(pred_mask)
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        # if self.mask and (not isTrain):
        #     pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
        #                                                     3, 0, 1)  # [1, 1, H, W]
        #     ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        # return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}


class Renderer_super(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_super, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim= args.feature_channels).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.feature_channels).to(args.device)

        self.unet = UNet(args).to(args.device)
        self.unet_color = UNet_color(args).to(args.device)
        self.unet_super = UNet_super(args).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, colors, ray, zbufs, gt, mask_gt, isTrain, xyz_o):
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

        H, W, C = colors.shape  # [H, W, 3*points_per_pixel]
        H, W, _ = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 1]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:]  # [H, W, 1]

        _o = o.unsqueeze(-2).expand(H, W, 1, 3)
        _dirs = dirs.unsqueeze(-2).expand(H, W, 1, 3)
        _cos = cos.unsqueeze(-2).expand(H, W, 1, 3)

        if self.use_crop and isTrain:
            # zbufs
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            # color
            colors_pad = self.pad_w(colors.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            
            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, colors_pad, gt_pad, mask_gt, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, colors_pad, gt_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)

            _, _, H, W = cat_img.shape
            o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            colors = cat_img[0, 7:31].permute(1, 2, 0)
            gt_crop = cat_img[0, 31:34].permute(1, 2, 0)
            gt = gt_crop

            
            if mask_gt is not None:
                mask_gt = cat_img[0, 34].permute(1, 2, 0)
                zbufs = cat_img[0, 15:].permute(1, 2, 0)
            else:
                zbufs = cat_img[0, 34:].permute(1, 2, 0)

            # _o = ray[:H, :W, :3].unsqueeze(-2)
            _o = o_crop.unsqueeze(-2).expand(H, W, 1, 3)
            _dirs = dirs_crop.unsqueeze(-2).expand(H, W, 1, 3)
            _cos = cos_crop.unsqueeze(-2).expand(H, W, 1, 3)

            del cat_img, ray_pad, gt_pad, zbufs_pad
            torch.cuda.empty_cache()

        ############################################
        
        # feature map of zbuf
        feature_map_zbufs=[]
        for i in range(zbufs.shape[-1]):
            zbuf = zbufs[..., i].unsqueeze(-1)  # [H, W, 1]

            if isTrain:
                pix_mask = zbuf > 0.2  # [H, W, 1]
            else:
                pix_mask = zbuf > 0

            o = _o[pix_mask]  # occ_point 3
            dirs = _dirs[pix_mask]  # occ_point 3
            cos = _cos[pix_mask]  # occ_point 1
            zbuf = zbuf.unsqueeze(-1)[pix_mask]  # occ_point 1

            if self.xyznear:
                xyz_near = o + dirs * zbuf / cos  # occ_point 3
            else:
                xyz_near = xyz_o[zbuf.squeeze(-1).long()]

            feature = self.mlp(xyz_near, dirs)  # occ_point 3
            del xyz_near, zbuf

            feature_map = torch.zeros([H, W, 1, self.dim], device=zbufs.device)
            feature_map[pix_mask] = feature  # [400, 400, 1, self.dim]
            feature_map_zbufs.append(feature_map.permute(2, 3, 0, 1)) # [400, 400, self.dim, burst]
            del feature_map 

        # [burst, 4 (self.dim), H, W]
        feature_map_zbufs = torch.cat(feature_map_zbufs, dim=0)

        del _o, _dirs, _cos
        torch.cuda.empty_cache()

        ################################################
        
        # feature map of color
        feature_map_color=[]
        color = []
        for i in range(self.points_per_pixel):
            indices = [3*i, 3*i+1, 3*i+2]
            color.append(colors[:, :, indices].unsqueeze(0))
        colors = torch.cat(color, dim=0) # [burst, H, W, 3]
        for i in range(colors.shape[0]):
            color = colors[i].unsqueeze(0).permute(0, 3, 1, 2)  # [3, H, W]
            feature_map_color.append(self.unet_super(color)) # [burst, self.dim, H, W]
        # [burst, self.dim, H, W]
        feature_map_color = torch.cat(feature_map_color, dim=0)

        ################################################

        # combine the feature map
        # [burst, 2*self.dim, H, W]
        feature_map = torch.cat((feature_map_zbufs, feature_map_color), dim = 1)

        # [burst, 8, H, W]
        pred_mask = self.mpn(feature_map)

        # initial alpha-blending
        alpha = torch.cumprod((1 - pred_mask),dim=1)[:,:-1,:,:]
        ones = (torch.ones_like(pred_mask).to(pred_mask.device)[:,0,:,:]).unsqueeze(1)
        alpha = torch.cat((ones,alpha), dim=1)
        fuse_rdmp = feature_map.mul(pred_mask).mul(alpha)
        fuse_rdmp =  torch.sum(fuse_rdmp, dim=1).unsqueeze(0)

        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        # if self.mask and (not isTrain):
        #     pix_mask = pix_mask.int().unsqueeze(-1).permute(2, 3, 0, 1)  # [1, 1, H, W]
        #     ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        # return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}


if __name__ == '__main__':
    device = torch.device("cuda")
    parser = config_parser()
    args = parser.parse_args()
    print(args)
