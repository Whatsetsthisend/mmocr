from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import init_detector

from mmocr.apis.inference import model_inference
from mmocr.datasets import build_dataset  # noqa: F401
from mmocr.models import build_detector  # noqa: F401


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('save_path', help='Folder to save visualized images.')
    parser.add_argument(
        '--images',
        nargs='+',
        help='Image files to be predicted with batch mode, '
        'separated by space, like "image_1.jpg image2.jpg".')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')

    # build the model from a config file and a checkpoint file
    model = init_detector('../configs/textdet/panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                          '../configs/textdet/panet/panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth',
                          device='cuda:0')

    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline
    import cv2
    img = cv2.imread('./images/22.png')
    out_file = './demo_text_det_pred.jpg'
    # test multiple images
    result = model_inference(model, img, batch_mode=True)
    print(f'results: {result}')


    out_file = './demo_text_det_pred.jpg'

    # show the results
    img = model.show_result(
        img, result, out_file=out_file, show=False)
    if args.imshow:
        mmcv.imshow(img, f'predicted results ({img})')


if __name__ == '__main__':
    main()
