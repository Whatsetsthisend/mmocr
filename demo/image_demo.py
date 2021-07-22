from argparse import ArgumentParser

import mmcv
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('img', default='./images/1.png', type=str, help='Image file.')
    parser.add_argument('config', default='../configs/textrecon/seg/seg_r31_1by16_fpnocr_academic.py',
                        help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.',
                        default='../configs/textrecon/seg/seg_r31_1by16_fpnocr_academic-72235b11.pth')
    parser.add_argument('out_file', help='Path to save visualized image.', default='./demo_text_det_pred.jpg')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.',
        default=True)
    # args = parser.parse_args()
    # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # model = init_detector('../configs/textrecog/robust_scanner/robustscanner_r31_academic.py',
    #                       '../configs/textrecog/robust_scanner/robustscanner_r31_academic-5f05874f.pth', device='cuda:0')
    # model = init_detector('../configs/textrecog/sar/sar_r31_parallel_decoder_academic.py',
    #                       '../configs/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth', device='cuda:0')
    # model = init_detector('../configs/textrecog/sar/sar_r31_sequential_decoder_academic.py',
    #                       '../configs/textrecog/sar/sar_r31_sequential_decoder_academic-d06c9a8e.pth', device='cuda:0')

    # model = init_detector('../configs/textrecog/nrtr/nrtr_r31_1by8_1by4_academic.py',
    #                       '../configs/textrecog/nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth', device='cuda:0')

    model = init_detector('../configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                          '../configs/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth', device='cuda:0')

    # model = init_detector('../configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py',
    #                       '../configs/textrecog/seg/seg_r31_1by16_fpnocr_academic-72235b11.pth', device='cuda:0')

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    print(model.modules())
    # model
    img = cv2.imread('./images/22.png')
    out_file = './demo_text_det_pred.jpg'
    # test a single image
    result = model_inference(model, img)
    print(f'result: {result}')

    print(result.shape)
    # show the results
    img = model.show_result(
        img, result, out_file=out_file, show=False)

    if img is None:
        img = mmcv.imread(img)

    mmcv.imwrite(img, out_file)
    if True:
        mmcv.imshow(img, 'predicted results')


if __name__ == '__main__':
    main()
