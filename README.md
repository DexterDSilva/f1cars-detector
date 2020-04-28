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


## Creating image dataset
To create a reasonably clean set of images I downladed the images to my MBP and in a Finder window i just scrolled through the images
and deleted the ones that would appear confusing to the nn ie ones with two or more cars, model images, sketches etc

# All images are in colour. Therefore the test images should also be in colour.

See the f1cars-downloader.ipynb for info on downloading to a local directory\
The data file is f1cars_images.zip

The image directories can be uploaded to your Google drive

In the noteboks you will see my Google drive as 
'/content/drive/My Drive/fastai-v3/lesson2/f1cars/





