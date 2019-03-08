import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import numpy as np

from tqdm import tqdm
from dataset import DF314
from models.unet import create_model


def predict(model, folder_name, batch_size, mode='predict'):
    print('predict start, load data from %s' % folder_name)
    loader = DF314(folder_name, mode=mode, batch_size=batch_size)

    for _, (inputs, paths) in enumerate(tqdm(loader)):
        outputs = model.predict(inputs)

        for i, image_name in enumerate(paths):
            p_mask = np.argmax(outputs[i]).astype(np.uint8)
            np.save('{}/predicted/channel2/category/{}'.format(folder_name, image_name), p_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='keras.model', help='path to model folder')
    arg('--batch-size', type=int, default=10)
    arg('--root', default='../input/df314/test', help='The root of train data folder')
    arg('--mode', type=str, default='test')

    args = parser.parse_args()

    num_classes = 8
    model = create_model(64, 4000, 1, num_classes=num_classes)
    model.load_weights(args.model_path)
    print('model loaded')
    print('saved to {}/predicted/channel2/category'.format(args.root))
    saved_path = '{}/predicted/channel2/category'.format(args.root)
    if os.path.exists(saved_path) is not True:
        os.mkdir(saved_path)

    predict(model, args.root, args.batch_size, mode=args.mode)
    print('Predicted done!')