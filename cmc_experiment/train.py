from model import *
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import mlflow
import glob
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

parser = argparse.ArgumentParser(description="CMC gaussian U-Net")
parser.add_argument("--exp_name", type=str,
                    default="CMC gaussian U-Net", help="exp_name")
parser.add_argument("-d", type=str,
                    default="train_gauss", help="choose dataset")
parser.add_argument("-g", type=str,
                    default="0", help="choose gpu")
parser.add_argument("-lr", type=float,
                    default=1e-5, help="learning rate")
parser.add_argument("-e", type=int,
                    default=100, help="epochs")
parser.add_argument("-b", type=int,
                    default=4, help="batch size")
parser.add_argument("-s", type=int,
                    default=256, help="image size")
parser.add_argument("--seed", type=int,
                    default=0, help="seed")
parser.add_argument("-m", type=str,
                    default="unet", help="model")
parser.add_argument("--std", type=int,
                    default=10, help="gauss std")
parser.add_argument("--opt", type=str,
                    default="Adam", help="optimizer")
parser.add_argument("-p", type=str,
                    default=None, help="pretrained weight")
args = parser.parse_args()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Fix numpy seed
np.random.seed(0)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                   image_color_mode = "rgb", mask_color_mode = "rgb",
                   target_size = (256, 256), seed = 0):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = img/255
        mask = mask/255
        yield img, mask
        
def main():
    dir_name = "h5/{}".format(args.d)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if args.m=="unet":
        model = unet(input_size = (args.s, args.s, 3))
        mask_color_mode = "rgb"
        loss = "categorical_crossentropy"
    elif args.m=="unet_bin":
        model = unet_bin(input_size = (args.s, args.s, 3))
        mask_color_mode = "grayscale"
        loss = "binary_crossentropy"
    else:
        print("ERROR!! args.m is not valid.")
        return 0
    
    if args.p is not None:
        model.load_weights(args.p)
        print("Model weight {} is loaded.".format(args.p))
    model.summary()
    
    # data augumentation
    train_gen_args = dict(horizontal_flip = True,
                        vertical_flip = True,
                        rotation_range = 45,
                        zoom_range = [0.95, 1.05],
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        fill_mode = "constant",
                        cval = 0)
    
    # generator of training data and validation data
    trainGene = trainGenerator(batch_size = args.b,
                            train_path = args.d,
                            image_folder = "image",
                            mask_folder = "label",
                            aug_dict = train_gen_args,
                            mask_color_mode = mask_color_mode,
                            target_size = (args.s, args.s),
                            seed = args.seed)
    
    if args.opt=="Adam":
        optimizer = Adam(lr = args.lr)
    elif args.opt=="SGD":
        optimizer = SGD(lr = args.lr)
    else:
        print("ERROR!! args.opt is not valid.")
        return 0
        
    model.compile(optimizer = optimizer, loss = loss)

    steps_per_epoch = len(glob.glob(args.d+"/image/*"))//args.b

    model_checkpoint = ModelCheckpoint("{}/{}.h5".format(dir_name, args.m), monitor="loss", verbose=1, save_best_only=True)
    history = model.fit_generator(trainGene,
                                  steps_per_epoch = steps_per_epoch,
                                  epochs = args.e,
                                  callbacks=[model_checkpoint])
    
    fig = plt.figure()
    plt.plot(range(1, args.e+1), history.history["loss"], label="training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig("loss.png")
    
    mlflow.set_experiment(args.exp_name)
    with mlflow.start_run() as run:
        mlflow.log_params({"learning_rate": args.lr,
                            "epoch_num": args.e,
                            "batch_size": args.b,
                            "image_size": args.s,
                            "seed": args.seed,
                            "model": args.m,
                            "gauss std": args.std,
                            "optimizer": args.opt,
                            "loss": loss})
        mlflow.log_metrics({"train_loss": history.history["loss"][-1]})
        mlflow.log_artifact("loss.png")

if __name__ == "__main__":
    print(args)
    main()
