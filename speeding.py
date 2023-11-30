# https://github.com/feinanshan/FANet/blob/master/Testing/speeding.py


import time
import torch
from utils import config_parser
from model.renderer import Renderer
from raw_bpcr_model.renderer import BPCR

def run(model,size,name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():

        input = torch.rand(size).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(100):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts-start_ts #t_cnt + (end_ts-start_ts)
    print("=======================================")
    print("Model Name: "+name)
    print("FPS: %f"%(100/t_cnt))



parser = config_parser()
args = parser.parse_args()
renderer = Renderer(args)

zbuf = torch.rand((800, 800, args.points_per_pixel)).to(args.device)
ray = torch.rand((800, 800, 7)).to(args.device)

renderer.eval()
t_cnt = 0.0
with torch.no_grad():
    torch.cuda.synchronize()
    x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start_ts = time.time()
    for i in range(100):
        x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    torch.cuda.synchronize()
    end_ts = time.time()
    t_cnt = end_ts-start_ts #t_cnt + (end_ts-start_ts)
print("=======================================")
print("Ours FPS: %f"%(100/t_cnt))



renderer = BPCR(args)

zbuf = torch.rand((800, 800, 1)).to(args.device)
ray = torch.rand((800, 800, 7)).to(args.device)

renderer.eval()
t_cnt = 0.0
with torch.no_grad():
    torch.cuda.synchronize()
    x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start_ts = time.time()
    for i in range(100):
        x = renderer(zbuf, ray, gt=None,
                  mask_gt=None, isTrain=False, xyz_o=None)
    torch.cuda.synchronize()
    end_ts = time.time()
    t_cnt = end_ts-start_ts #t_cnt + (end_ts-start_ts)
print("=======================================")
print("BPCR FPS: %f"%(100/t_cnt))
