# FADA-Pytorch
pytorch implement for the NIPS 2017 paper [Few-Shot Adversarial Domain Adaptation](https://arxiv.org/abs/1711.02536)


### Install dependencies

* This code has been tested on  Python 3.6. pytorch 0.4. 
* Install [PyTorch, torchvision](http://pytorch.org/) and [SciPy](https://www.scipy.org/): `pip install -r requirements.txt`


### Set up the  dataset

* Run ` python main.py `and  get MNIST and SVHN dataset in ./data/

### Train the model

* Run `python main.py`. 
  * Step 1 refers to pretrain model using MNIST dataset
  * Setp 2 refers to pretrain DCD using G1,G2,G3,G4.
  * Step 3 refers to train g,h and DCD iteratively.

### Evaluate

* Using 7 shot in target domain , training step 3 after 30 epoches ,I got the same result 47% acc as shown in papers
![results](https://github.com/1025616969/FADA-Pytorch/blob/master/screenshot.png)

