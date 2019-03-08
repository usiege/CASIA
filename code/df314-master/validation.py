import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import numpy as np

from tqdm import tqdm
from dataset import DF314
from models.unet import create_model
from utils import load_csv


def predict(model, folder_name, batch_size, mode='predict'):
    print('predict start, load data from %s' % folder_name)
    loader = DF314(folder_name, mode=mode, batch_size=batch_size)

    counts = np.zeros((7, 2), dtype=np.uint64)

    for _, (inputs, idx, paths) in enumerate(tqdm(loader)):
        outputs = model.predict(inputs)

        for i, image_name in enumerate(paths):
            p_mask = np.argmax(outputs[i], axis=2).astype(np.uint8)
            categories = p_mask.reshape(-1)
            ids = idx[i].reshape(-1)

            valid_idx = (ids > 0) & (categories > 0)
            categories = categories[valid_idx]
            ids = ids[valid_idx]

            groud_truth = load_csv('{}/category/{}.csv'.format(folder_name, image_name[:-4]))
            groud_truth = groud_truth.astype(np.uint8).reshape(-1)

            predicted = np.zeros(groud_truth.shape, dtype=np.uint8)
            for i, j in enumerate(ids):
                predicted[j] = categories[i]

            for i in range(1, 8):
                gt = (groud_truth == i)
                pt = (predicted == i)
                counts[i-1, 0] += np.sum(np.logical_and(gt, pt))
                counts[i-1, 1] += np.sum(np.logical_or(gt, pt))

    ious = 1.0 * counts[:, 0] / counts[:, 1]
    for i, iou in enumerate(ious):
        print('iou{}: {}'.format(i + 1, iou))
    print('mean iou: {}'.format(np.mean(iou)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='keras.model', help='path to model folder')
    arg('--batch-size', type=int, default=10)
    arg('--root', default='../input/df314/training', help='The root of train data folder')
    arg('--mode', type=str, default='predict')

    args = parser.parse_args()

    num_classes = 8
    model = create_model(64, 4000, 1, num_classes=num_classes)
    model.load_weights(args.model_path)
    print('model loaded')

    predict(model, args.root, args.batch_size, mode=args.mode)
    print('Predicted done!')