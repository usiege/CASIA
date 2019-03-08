import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from dataset import DF314
from loss import focal_loss
from models.unet import create_model
from metrics import iou_metric


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--root', default='../input/df314/training', help='The root of train data folder')
    arg('--log-dir', default='log', help='the train output folder')
    arg('--last-model-name', default='keras.model', help='the name of model for loading')
    arg('--save-model-name', default='keras.model', help='the name of model to save')
    arg('--batch-size', type=int, default=8)
    arg('--epochs', type=int, default=100)
    arg('--lr', type=float, default=0.01)
    args = parser.parse_args()

    lr = args.lr
    saved_model = args.save_model_name
    last_model = args.last_model_name

    num_classes = 8
    train_loader = DF314(args.root, batch_size=args.batch_size, num_classes=num_classes, fold=True)

    model = create_model(64, 4000, 1, num_classes=num_classes)
    if os.path.exists(last_model):
        model.load_weights(last_model)
        print('Last trained weight loaded')

    # add callback
    model_checkpoint = ModelCheckpoint(saved_model, monitor='val_iou_metric_func', mode='max', save_best_only=True,
                                       verbose=1, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_iou_metric_func', factor=0.5, patience=8, min_lr=0.0001, verbose=1)

    optimizer = SGD(lr=lr)
    loss_weight = [0, 2., 2., 1.2, 1.2, 2., 3., 2.]
    loss_func = focal_loss(num_classes=num_classes, loss_weight=loss_weight)
    valid_func = iou_metric(num_classes=num_classes)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=[valid_func])
    model.fit_generator(train_loader, epochs=50, max_queue_size=10, workers=0, verbose=1,
                        validation_data=train_loader,
                        validation_steps=train_loader.get_validation_step(),
                        callbacks=[model_checkpoint, reduce_lr])


if __name__ == '__main__':
    main()
