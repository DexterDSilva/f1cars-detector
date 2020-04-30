# f1cars-detector
Fastai based example to detect current 2019-2020 cars
Detects a car image belongs to either of\
Ferrari, Mercedes, RedBull, RacingPoint, McLaren,Renault,ToroRosso,Williams, Haas, Sauber
Note: These are the cars racing in the 2017 to 2019 season. 

![alt text](https://github.com/DexterDSilva/f1cars-detector/blob/master/fp-1.png "Selection")

# RESULT:

![alt text](https://github.com/DexterDSilva/f1cars-detector/blob/master/fp-3.png "Results")





> Setup
> Macbook Pro 2015 for downloading and sorting images 
> Upload images to Google drive 
> Use Google Colab with GPU for training
> Chose ResNet50 after trial and error. Choice confirmed by [Pyimage](https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)

The notebook is based on Jeremy Howard's course-v3/lesson2-download
from https://github.com/fastai/course-v3

Nb Classes = 10\
Nb Images/class: \
  108  images in ferrari \
  102  images in  mercedes \
  118  images in  redbull \
  117  images in  mclaren \
  83  images in  racingpoint \ 
  95  images in  renault \
  120  images in  williams \
  98  images in  tororosso \
  123  images in  haas \
  108  images in  sauber\


# Creating image dataset
To create a reasonably clean set of images I downladed the images to my MBP and in a Finder window i just scrolled through the images
and deleted the ones that would appear confusing to the nn ie ones with two or more cars, model images, sketches etc

# All images are in colour. Therefore the test images should also be in colour.

See the f1cars-downloader.ipynb for info on downloading to a local directory\
The data file is f1cars_images.zip

The image directories can be uploaded to your Google drive

In the noteboks you will see my Google drive as 
'/content/drive/My Drive/fastai-v3/lesson2/f1cars/

# The Resnet model
ResNet was first introduced by He et al. in their seminal 2015 paper, Deep Residual Learning for Image Recognition — that paper has been cited an astonishing 43,064 times! [Pyimage](https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)

A follow-up paper in 2016, Identity Mappings in Deep Residual Networks, performed a series of ablation experiments, playing with the inclusion, removal, and ordering of various components in the residual module, ultimately resulting in a variation of ResNet that:

- [x] Is easier to train
- [x] Is more tolerant of hyperparameters, including regularization and initial learning rate
- [x] Generalizes better

# Error handling
I used Google colab and sometime after this training the torch version was upgraded which meant a lot of errors and inability to use augmentation.

Use 
> !pip install “torch==1.4” “torchvision==0.5.0”


from fastai import __version__
fastai.__version [op: 1.0.60]

from fastai import *
torch.__version__  [op: 1.4.0]



Thanks to [fastai forum petrovi4](https://forums.fast.ai/t/lesson-2-getting-error-with-creating-imagedatabunch-from-folder/64137/25)







