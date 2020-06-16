from glob import glob
from skimage.transform import resize
from skimage import io, img_as_ubyte, img_as_bool
from skimage.segmentation import find_boundaries, mark_boundaries
import numpy as np
import os
from tqdm import *
from segmentation_net import PangNet, Unet, UnetPadded, BatchNormedUnet, ExampleDatagen
from segmentation_net.utils import expend
import matplotlib.pylab as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def LoadRGB(name, mask=False):
    img = io.imread(name)
    if not mask:
        img = img[:,:,0:3]
    else:
        img[img < 125] = 0
        img[img > 0] = 1
    return img

def find_mask(path):
    return path.replace('raw', 'lbl')

def plot_with_annotation(raw_name, title=None):
    rgb = LoadRGB(raw_name)
    mask = LoadRGB(find_mask(raw_name), mask=True) 
    plot_overlay(rgb, mask, title)


def FindRGB(name, train=True):
    sigle = os.path.basename(name).split('_')[0]
    if train:
        rgb = name.replace('Annotations', 'Smaller{}').format(sigle)
    else:
        rgb = name.replace('AnnotationsTest', 'Smaller{}').format(sigle)
    png = rgb.replace('nii.gz', 'png')
    return png
def plot(name):
    rgb = LoadRGB(name)
    plt.imshow(rgb)
    plt.show()

def AddContours(image, label, color = (0, 1, 0)):
    res = mark_boundaries(image, label, color=color)
    res = img_as_ubyte(res)
    return res

def plot_triplet(a, b, c, title=None):
    fig, axes = plt.subplots(ncols=3, figsize=(25,8))
    axes[0].imshow(a)
    axes[0].axis('off')
    axes[1].imshow(b)
    axes[1].axis('off')
    axes[2].imshow(c)
    if title is not None:
        plt.suptitle(title, fontsize=16)
    axes[2].axis('off')

def plot_overlay(a, b, title=None):
    plot_triplet(a, b, AddContours(a, b), title)

def plot_slide(img_path, title=None):
    mask_path = img_path.replace('Slide', 'GT')
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    plot_overlay(img, mask, title)

# def plot_annotation(mask_name, title=None, train=False):
#     rgb = LoadRGB(FindRGB(mask_name, train))
#     mask = LoadNiiGz(mask_name) 
#     plot_overlay(rgb, mask, title)

def Predict(model, rgb):
    #import pdb; pdb.set_trace()
    original_shape = rgb.shape[0:2]
    resized_rgb = resize(rgb, (512, 512), order=0, preserve_range=True).astype(rgb.dtype)
    resized_mask = model.predict(resized_rgb)
    mask = resize(resized_mask['predictions'], (original_shape), order=0, preserve_range=True).astype('uint8')
    return mask

def RandomImagePick(path, n_copies=1):
    files = glob(path)
    for _ in range(n_copies):
        f = np.random.choice(files, replace=False)
        yield f, LoadRGB(f)

class TissueGenerator(ExampleDatagen):
    def __init__(self, path, size=(512, 512), train=True):
        files = glob(path)
        self.size = size
        self.length = len(files) * 4
        self.indices = [(f, i) for i in range(4) for f in files]
        self.train = train
        self.dic = {(f, i): self.create_couple(f, i) for f, i in self.indices}
        self.current_iter = 0
    def load_img(self, image_name):
        img = LoadRGB(image_name)
        img = resize(img, self.size, order=0, 
                     preserve_range=True, mode='reflect', 
                     anti_aliasing=True).astype(img.dtype)
        # img = img.astype('float32')
        # img = img / 255.
        return img
    def load_mask(self, image_name):
        img = LoadRGB(find_mask(image_name), mask=True)
        img = resize(img, self.size, order=0, 
                     preserve_range=True, mode='reflect',
                     anti_aliasing=True).astype(img.dtype)
        return img

def GetCoord(integer):
    if integer == 0:
        x, x_e = 0, 250
    elif integer == 1:
        x, x_e = 250, 500
    elif integer == 2:
        x, x_e = 500, 750
    elif integer == 3:
        x, x_e = 750, 1000
    return x, x_e

class CellImageGenerator(ExampleDatagen):
    def __init__(self, path, pattern, size=(1000, 1000), train=True):
        files = glob(path)
        if pattern == "train":
            files = [f for f in files if 'test' or 'validation' in f]
        elif pattern == "validation":
            files = [f for f in files if 'validation' in f]
        elif pattern == "test":
            files = [f for f in files if 'test' in f]
        else:
            print("Not normal")
        self.size = size
        self.length = len(files) * 16
        self.indices = [(f, i) for i in range(16) for f in files]
        self.train = train
        self.dic = {(f, i): self.create_couple(f, i) for f, i in self.indices}
        self.current_iter = 0
    def load_img(self, image_name):
        img = LoadRGB(image_name)
        return img
    def load_mask(self, image_name):
        mask_name = image_name.replace('Slide', 'GT')
        mask = io.imread(mask_name)
        mask[mask > 0] = 1
        return mask
    def quarter_image(self, image, integer):
        abscisse = integer / 4
        ordonnee = integer - abscisse * 4
        x, x_e = GetCoord(abscisse)
        y, y_e = GetCoord(ordonnee)
        return image[x:x_e, y:y_e]

class CellImageGeneratorTest(CellImageGenerator):
    def __init__(self, path, pattern, size=(1000, 1000), train=True):
        files = glob(path)
        if pattern == "train":
            files = [f for f in files if 'test' or 'validation' in f]
        elif pattern == "validation":
            files = [f for f in files if 'validation' in f]
        elif pattern == "test":
            files = [f for f in files if 'test' in f]
        else:
            print("Not normal")
        self.size = size
        self.length = len(files) * 1
        self.indices = [(f, i) for i in range(1) for f in files]
        self.train = train
        self.dic = {(f, i): self.create_couple(f, i) for f, i in tqdm(self.indices)}
        self.current_iter = 0
    def quarter_image(self, image, integer):
        return image
    def load_mask(self, image_name):
        mask_name = image_name.replace('Slide', 'GT')
        mask = io.imread(mask_name)
        return mask

class TissueGeneratorUnet(TissueGenerator):
    def expand(self, image):
        x, y = 92, 92
        new_img = expend(image, x, y)
        return(new_img)

    def create_couple(self, image_name, integer):
        img = self.expand(self.load_img(image_name))
        mask = self.expand(self.load_mask(image_name))

        return (self.quarter_image(img , integer), 
                self.quarter_image(mask, integer))

    def quarter_image(self, image, integer):
        if integer == 0:
            x, y = 0, 0
            x_e, y_e = 256 + 184, 256 + 184
        elif integer == 1:
            x, y = 0, 256
            x_e, y_e = 256 + 184, 512 + 184
        elif integer == 2:
            x, y = 256, 0
            x_e, y_e = 512 + 184, 256 + 184
        elif integer == 3:
            x, y = 256, 256
            x_e, y_e = 512 + 184, 512 + 184
        return image[x:x_e, y:y_e]

from sklearn.metrics import f1_score

def F1(annotation, prediction):
    return f1_score(annotation.flatten(), prediction.flatten())

def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for y in range(0, image.shape[0] - windowSize[0] + stepSize, stepSize):
        for x in range(0, image.shape[1] - windowSize[1] + stepSize, stepSize):
            # yield the current window
            res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            change = False
            if res_img.shape[0] != windowSize[1]:
                y = image.shape[0] - windowSize[1]
                change = True
            if res_img.shape[1] != windowSize[0]:
                x = image.shape[1] - windowSize[0]
                change = True
            if change:
                res_img = image[y:y + windowSize[1], x:x + windowSize[0]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)

def slidding_window_predict(img, mod):
    rgb_shape_x = img.shape[0] - 184
    rgb_shape_y = img.shape[1] - 184
    res = np.zeros(shape=(rgb_shape_x, rgb_shape_y), dtype='uint8')
    res_prob = np.zeros(shape=(rgb_shape_x, rgb_shape_y), dtype='uint8')
    for x, y, w_x, h_y, slide in sliding_window(img, 212, (396, 396)):
        tensor = np.expand_dims(slide, 0)
        feed_dict = {mod.rgb_ph: tensor,
                     mod.is_training: False}
        tensors_names = ["predictions", "probability"]
        tensors_to_get = [mod.predictions, mod.probability]
        
        tensors_out = mod.sess.run(tensors_to_get,
                                    feed_dict=feed_dict)
        res[y:(y+212), x:(x+212)] = tensors_out[0][0]
        res_prob[y:(y+212), x:(x+212)] = tensors_out[1][0,:,:,0]
    res[res > 0] = 255
    all_tensors = [res, res_prob]
    out_dic = {}
    for name, tens in zip(tensors_names, all_tensors):
        out_dic[name] = tens
    return out_dic
    
def unet_pred(img, mod):
    prev_shape = img.shape[0:2]
    img = resize(img, (512, 512), order=0, 
                 preserve_range=True, mode='reflect', 
                 anti_aliasing=True).astype(img.dtype)
    unet_img = expend(img, 92, 92)
    #mask = model.predict(rgb, mean=mean_array)['predictions'].astype('uint8')
    dic_mask = slidding_window_predict(unet_img, mod)
    dic_mask['predictions'][dic_mask['predictions'] > 0] = 255
    dic_mask['predictions'] = img_as_bool(resize(dic_mask['predictions'], prev_shape, order=0, 
                                     preserve_range=True, mode='reflect', 
                                     anti_aliasing=True))
    dic_mask['probability'] = resize(dic_mask['probability'], prev_shape, order=0, 
                                     preserve_range=True, mode='reflect', 
                                     anti_aliasing=True).astype(dic_mask['probability'].dtype)
    return dic_mask
      
# defining the l1 loss
LOSSES_COLLECTION = '_losses'
def l1_loss(tensor, weight=1.0, scope=None):
    """
    Define a L1Loss, useful for regularize, i.e. lasso.
    Args:
        tensor: tensor to regularize.
        weight: scale the loss by this factor.
        scope: Optional scope for name_scope.
    Returns:
        the L1 loss op.
    """
    with tf.name_scope(scope, 'L1Loss', [tensor]):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
        tf.add_to_collection(LOSSES_COLLECTION, loss)
        return loss



# ploting curves

def plot_curves(name, dic, keep=None):
    res = pd.DataFrame()
    for k in dic.keys():
        if name in k:
            if keep is None:
                test = dic[k]['test']
                mini_tab = test[['f1_score']]
                mini_tab['step'] = mini_tab.index
                mini_tab['Category'] = k
                res = pd.concat([res, mini_tab], axis=0)
            else:
                if k in keep:
                    test = dic[k]['test']
                    mini_tab = test[['f1_score']]
                    mini_tab['step'] = mini_tab.index
                    mini_tab['Category'] = k
                    res = pd.concat([res, mini_tab], axis=0)

    res = res.fillna(0)
    plt.figure(figsize=(20,10))
    sns.pointplot(x="step", y="f1_score", hue="Category", data=res)
    plt.show()
    return res


def id_(x):
    return x
def plot_quadoverlay(best_model, test_img, mean_array, n, aux_func=id_):
    fig, axes = plt.subplots(nrows=4, ncols=n, figsize=(16,16), sharex=True, sharey=True)
    for key in best_model.keys():
        model = return_good_object(key, best_model[key], mean_array)
        k = 0
        for name, rgb in test_img:
            if 'Unet' == key:
                row = 1
                mask = Predict(model, rgb)
                rgb = rgb.astype('uint8')
            elif 'BatchNormedUnet' in key:
                mask = unet_pred(rgb, model)['predictions']
                row = 3
                axes[row, k].set_xlabel(os.path.basename(name), fontsize=12)
            elif 'UnetPadded' in key:
                row = 2
                mask = unet_pred(rgb, model)['predictions']
            elif 'PangNet' in key:
                row = 0
                mask = model.predict(rgb)['predictions'].astype('uint8')
            cleaned_mask = aux_func(mask)
            axes[row, k].imshow(AddContours(rgb, cleaned_mask))
            axes[row, k].get_xaxis().set_ticks([])
            axes[row, k].get_yaxis().set_ticks([])
            #axes[row, k].axis('off')
            if k == 0:
                axes[row, k].set_ylabel(key, fontsize=12)
            k += 1
            
def return_good_object(key1, key2, mean_array):
    num_channels = 3
    tmp = 'tmp'
    num_labels = 2 
    tensorboard = False
    displacement = 0
    if key1 == "PangNet":
        model = PangNet(image_size=(224, 224), log=os.path.join(tmp, key2), 
                       num_channels=num_channels, num_labels=num_labels,
                       seed=None,
                       verbose=0, displacement=displacement, mean_array=mean_array)
    
    elif key1 == "Unet":
        arch = int(key2.split('__')[2])
        model = Unet(image_size=(256, 256), log=os.path.join(tmp, key2), 
                     num_channels=num_channels, num_labels=num_labels,
                     seed=None,
                     verbose=0, n_features=arch, mean_array=mean_array)
    elif key1 == 'UnetPadded': 
        mod, lr, arch = key2.split('__')
        model = UnetPadded(image_size=(212, 212), log=os.path.join(tmp, key2), 
                           num_channels=num_channels, num_labels=num_labels,
                           seed=None,
                           verbose=0, n_features=int(arch), mean_array=mean_array)
    elif key1 == 'BatchNormedUnet':
        mod, lr, _ = key2.split('__')
        model = BatchNormedUnet(image_size=(212, 212), log=os.path.join(tmp, key2), 
                                num_channels=num_channels, num_labels=num_labels,
                                seed=None,
                                verbose=0, n_features=16, mean_array=mean_array)
    return model
def plot_validation(val_res, list_rgb, list_label, unet=False, n=5):
    for _ in range(len(list_rgb)):
        rgb = list_rgb[_].astype('uint8')
        if unet:
            rgb = rgb[92:-92, 92:-92]
        lab = list_label[_].astype('uint8')
        pred = val_res['predictions'][_].astype('uint8')
        b = AddContours(rgb, lab)
        c = AddContours(rgb, pred)
        score = val_res['f1_score'][_]
        plot_triplet(rgb, b, c, title="Comparaison RGB/GT/Prediction. f1_score: {}".format(score))
        if _ == n:
            break

def recap_tab(table, name):
    tab = table.copy()
    mod = 'Model'
    lr = 'Learning rate'
    wd = 'Weight decay'
    c = "Category"
    l = "Loss"
    bs = "Batch Size"
    arch = "N_features"
    if name == "PangNet":
        tab[mod], tab[lr], tab[wd], tab[l] = tab[c].str.split('__', 4).str
        tab = tab.drop(c, 1)
        tab = tab.groupby([mod, lr, wd, l]).max()
    elif name == "Unet" or name == "UnetPadded":
        tab[mod], tab[lr], tab[arch] = tab[c].str.split('__', 3).str
        tab = tab.drop(c, 1)
        tab = tab.groupby([mod, lr, arch]).max()
    elif name == "BatchNormedUnet":
        tab[mod], tab[lr], tab[bs] = tab[c].str.split('__', 3).str
        tab = tab.drop(c, 1)
        tab = tab.groupby([mod, lr, bs]).max()
        
    best = "__".join(tab["f1_score"].argmax())
    print("best model is: {}".format(best))
    return tab, best


def prep_val(dg_list, unet=False):
    val_rgb, val_lbl = [], []
    for DG in dg_list: # DG_Train, 
        for i in range(DG.length):
            rgb, lbl = DG.next()
            if not unet:
                val_rgb.append(rgb)
                val_lbl.append(lbl)
            else:
                val_rgb.append(rgb[22:-22, 22:-22])
                val_lbl.append(lbl[114:-114,114:-114])
    
    return val_rgb, val_lbl


def main():
    PATH_Train = './Data/Data_TissueSegmentation/Annotations/*.nii.gz'
    PATH_Test = './Data/Data_TissueSegmentation/AnnotationsTest/*.nii.gz'
    MeanFile = "./tmp/mean_file.npy"
    TrainRec = "./tmp/train.tfrecord"
    TestRec = "./tmp/test.tfrecord"
    from SegNetsTF import PangObject, CreateTFRecord 
    from SegNetsTF.TFRecord import ComputeMean

    DG_Train = TissueGenerator(PATH_Train)
    DG_Test  = TissueGenerator(PATH_Test, train=False)

    # for DG in [DG_Train, DG_Test]:
    #     for i in range(1):
    #         img, anno = DG.next()
    #         plot_overlay(img, anno)
    CheckOrCreate('./tmp')

    ComputeMean(MeanFile, [DG_Train, DG_Test])
    CreateTFRecord(TrainRec, [DG_Train])
    CreateTFRecord(TestRec, [DG_Test])

    N_CPUS = 8
    model = PangObject(TrainRec, 
                       TestRec, 
                       EPOCHS=1,
                       LEARNING_RATE=0.001, 
                       BATCH_SIZE=4, 
                       IMAGE_SIZE=(224, 224),
                       NUM_LABELS=2, 
                       NUM_CHANNELS=3, 
                       LRSTEP="3epoch", 
                       LOG="./tmp/small",
                       WEIGHT_DECAY=0.0005, 
                       NUM_PARALLEL_BATCHES=N_CPUS, 
                       MEAN_FILE=MeanFile)

    n = 1
    def RandomImagePick(n_copies=1):
        files = glob('./Data/Data_TissueSegmentation/SmallerWSI/*.png') + \
                glob('./Data/Data_TissueSegmentation/SmallerBio/*.png')
        for i in range(n_copies):
            yield LoadRGB(np.random.choice(files, replace=False))

    for rgb in RandomImagePick(n_copies=n):
        mask = Predict(model, rgb)
        plot_overlay(rgb, mask)

if __name__ == '__main__':
    main()
