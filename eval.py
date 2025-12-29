import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder, calculate_psnr
from data import test_dataloader
from utils import EvalTimer
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import time
import sys, scipy.io


def _eval(model, config):
    model_pretrained = os.path.join('results/', config.model_name, config.test_model)
    state_dict = torch.load(model_pretrained)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
    model.eval()
    with torch.no_grad():
        adder = Adder()
        psnr_adder = Adder()
        warmup_iters = 20
        ssim_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            _, _, h, w = input_img.size()
            pad_h = (16 - h % 16) if h % 16 != 0 else 0
            pad_w = (16 - w % 16) if w % 16 != 0 else 0
            input_padded = torch.nn.functional.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')
            measure_time = iter_idx >= warmup_iters
            if measure_time:
                torch.cuda.synchronize()
                tm_start = time.time()
            pred = model(input_padded)

            if measure_time:
                torch.cuda.synchronize()
                tm_end = time.time()
                elapsed = tm_end - tm_start
                adder(elapsed)
            else:
                elapsed = 0.0
            pred[config.num_subband] = pred[config.num_subband][:, :, :h, :w]
            p_numpy = pred[config.num_subband].squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)
            ssim = structural_similarity(p_numpy, in_numpy, win_size=3, data_range=1.0)

            save_name_R = os.path.join(config.result_dir, 'test_' + '%04d' % (iter_idx + 1) + '.png')

            pred[config.num_subband] = torch.clamp(pred[config.num_subband], 0, 1)
            result = F.to_pil_image(pred[config.num_subband].squeeze(0).cpu(), 'RGB')
            result.save(save_name_R)

            psnr_adder(psnr)
            ssim_adder(ssim)
            if measure_time:
                print('%d iter PSNR: %.2f time: %.4f' % (iter_idx + 1, psnr, elapsed))
            else:
                print(f'Warm-up iteration {iter_idx + 1}: iteration (time not recorded)')

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f dB' % (ssim_adder.average()))
        print("Average time (excluding warm-up): %.4f" % adder.average())