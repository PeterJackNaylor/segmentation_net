
import tensorflow as tf
from segmentation_net import SegmentationNet, Unet, UnetPadded, BatchNormedUnet
from segmentation_net import create_tfrecord, compute_mean
from HelperDemo import TissueGenerator, TissueGeneratorUnet
import os
def main():

    PATH_Train = '../Data/Data_TissueSegmentation/Annotations/*.nii.gz'
    PATH_Test = '../Data/Data_TissueSegmentation/AnnotationsTest/*.nii.gz'

    # preparing data
    ## for the standard segmentation networks 
    MeanFile = "./tmp/mean_file.npy"
    ## for the standard segmentation networks (size(input) - 184 == size(output))
    TrainRec_unet = "./tmp/train_unet.tfrecord"
    TestRec_unet = "./tmp/test_unet.tfrecord"



    DG_Train_unet = TissueGeneratorUnet(PATH_Train)
    DG_Test_unet  = TissueGeneratorUnet(PATH_Test, train=False)

    mean_array = compute_mean(MeanFile, [DG_Train_unet, DG_Test_unet])
    create_tfrecord(TrainRec_unet, [DG_Train_unet])
    create_tfrecord(TestRec_unet, [DG_Test_unet])

    num_channels = 3 # because it is rgb
    num_labels = 2 # because we have two classes: object, background
    tensorboard = False # to not keep results in a tensorboard
    verbose = 2 # to not print anything (0, 1, 2 levels)
    displacement = 0 # because size(input) == size(output)

    ## training:
    lr_procedure = "10epoch" # the decay rate will be reduced every 5 epochs
    batch_size = 4 # batch size for the
    decay_ema = 0.9999 # 
    k = 0.96 # exponential decay factor
    n_epochs = 60 # number of epochs
    early_stopping = 20 # when to stop training, 20 epochs of non progression
    save_weights = True # if to store as final weight the best weights thanks to early stopping
    num_parallele_batch = 8 # number batch to run in parallel (number of cpu usually)
    restore = True # allows the model to be restored at training phase (or re-initialized)

    wd = 0.00005
    loss = tf.nn.l2_loss
    arch = 16 
    lr = 0.001
    bs = 2
    LOG = os.path.join('tmp', 'unetbatchnorm__{}__{}').format(lr, bs)

    model = BatchNormedUnet(image_size=(212, 212), log=LOG, 
                            num_channels=num_channels, num_labels=num_labels,
                            tensorboard=tensorboard, seed=None,
                            verbose=verbose, n_features=arch)

    dic = model.train(TrainRec_unet, TestRec_unet, learning_rate=lr,
                      lr_procedure=lr_procedure, weight_decay=wd, 
                      batch_size=bs, decay_ema=decay_ema, k=k, 
                      n_epochs=n_epochs, early_stopping=early_stopping, 
                      mean_array=mean_array, loss_func=loss, 
                      verbose=verbose, save_weights=save_weights, 
                      num_parallele_batch=num_parallele_batch,
                      log=LOG, restore=restore)
    model.sess.close()


if __name__ == '__main__':
    main()