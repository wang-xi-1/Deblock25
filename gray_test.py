import os.path  
import logging  
import numpy as np  
from datetime import datetime  
from collections import OrderedDict  
import torch  
import cv2  
from Utils import utils_logger  
from Utils import utils_image as util  
import requests  
import argparse

def main():  
    quality_factor_list = [10, 20, 30, 40]
    testset_name = 'Classic5'  # 'LIVE1_gray' 'Classic5' 'BSDS500_gray'
    n_channels = 1              # set 1 for grayscale image, set 3 for color image  
    show_img = False            # default: False  

    # Initialize paths as empty strings
    H_path = ''  # Path for GT test images
    E_path = ''  # Path for saving estimated images
    model_path = './weights/deblock_grayscale.pth'

    for quality_factor in quality_factor_list:  
        result_name = f"{testset_name}"  # Only using testset_name  
        E_path_full = os.path.join(E_path, result_name, str(quality_factor))  # E_path, for Estimated images  
        print('E_path', E_path_full)  
        util.mkdir(E_path_full)  

        if os.path.exists(model_path):  
            print(f'Loading model from {model_path}')  
        else:  
            print('Model path does not exist')  
            return  # Exit if the model path is invalid  

        logger_name = f"{result_name}_qf_{quality_factor}"  
        utils_logger.logger_info(logger_name, log_path=os.path.join(E_path_full, logger_name + '.log'))  
        logger = logging.getLogger(logger_name)  
        logger.info('--------------- quality factor: {:d} ---------------'.format(quality_factor))  

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        border = 0  

        # ----------------------------------------  
        # Load model  
        # ----------------------------------------  

        from Networks.network_dctmamba import DeblockNet
        parser = argparse.ArgumentParser()
        parser.add_argument('--globel_res', type=int, default=0)
        args = parser.parse_args()
        model = DeblockNet(args, n_channels, n_channels, base_channel=28, num_res=8).to(device)

        model.load_state_dict(torch.load(model_path), strict=True)  
        model.eval()  
        for k, v in model.named_parameters():  
            v.requires_grad = False  
        model = model.to(device)  
        logger.info('Model path: {:s}'.format(model_path))  

        test_results = OrderedDict()  
        test_results['psnr'] = []  
        test_results['ssim'] = []  
        test_results['psnrb'] = []  

        H_paths = util.get_image_paths(H_path)  
        for idx, img in enumerate(H_paths):  

            # ------------------------------------  
            # (1) img_L  
            # ------------------------------------  
            img_name, ext = os.path.splitext(os.path.basename(img))  
            logger.info('{:->4d}--> {:>10s}'.format(idx + 1, img_name + ext))  
            img_L = util.imread_uint(img, n_channels=n_channels)  

            if n_channels == 3:  
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)  
            _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])  
            img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)  
            if n_channels == 3:  
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)  
            img_L = util.uint2tensor4(img_L)

            img_L = img_L.to(device)
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()

            H, W = img_L.shape[-2:]
            img_L = torch.nn.functional.pad(img_L,
                                            (0, (16 - img_L.shape[-1] % 16) % 16, 0, (16 - img_L.shape[-2] % 16) % 16),
                                            mode='constant', value=0)

            # ------------------------------------  
            # (2) img_E  
            # ------------------------------------  
            img_E = model(img_L)
            img_E = img_E[..., :H, :W]

            img_E = util.tensor2single(img_E)  
            img_E = util.single2uint(img_E)

            # --------------------------------  
            # PSNR and SSIM, PSNRB  
            # --------------------------------  
            psnr = util.calculate_psnr(img_E, img_H, border=border)  
            ssim = util.calculate_ssim(img_E, img_H, border=border)  
            psnrb = util.calculate_psnrb(img_H, img_E, border=border)  
            test_results['psnr'].append(psnr)  
            test_results['ssim'].append(ssim)  
            test_results['psnrb'].append(psnrb)  
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name + ext, psnr, ssim, psnrb))

            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None  
            util.imsave(img_E, os.path.join(E_path_full, img_name + '.png'))  

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])  
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])  
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])  
        logger.info(  
            'Average PSNR/SSIM/PSNRB - {} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'.format(result_name + '_' + str(quality_factor), ave_psnr, ave_ssim, ave_psnrb))  


if __name__ == '__main__':  
    main()