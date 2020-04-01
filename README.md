

<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>

[Paper GAN](https://arxiv.org/pdf/1406.2661.pdf)<br />
[Paper HyperNet](https://arxiv.org/pdf/1905.02898.pdf)
<br /><br />
## BackGround

* This is a good test for using HyperNet idea on GAN
* Still, no sure how to use this idea on something interesting experiment. So far, the only thing we know is that GAN can use the weights from other Net and generate the images we want.
* 

<br /><br />
## Quick Start
* Run train.py and the result will be stored in output
* The hyper params are already set up in train.py.
* The parms used for HyperNet are set up in params.py. If you have a better computer, feel free to try different params.
* The number of trainning images can be set up in loadData.py. Just simply change num to any number of images you want to train

<br /><br />
## Version of Installment
#### Tensorflow 1.13.1
#### Numpy 1.18.2
#### Python 3.6.9  

<br /><br />
## Structure of Network  
* In fact, the leaky ReLu should switch place with BN, but that's it, LOL.
* the code is correct, so don't worry about it.

### Generator of GAN
 ![Image text](https://github.com/CuiJiali-CV/cGAN/raw/master/Generator.png)
### Discriminator of GAN
 ![Image text](https://github.com/CuiJiali-CV/cGAN/raw/master/Discriminator.png)
 <br /><br />
### The detail of HyperNet for generating weights 
#### Generate Latent Space of Weights ( called Codes)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/raw/master/Codes_generated.png)
#### Generate Weights ( Still, here we are using FC all the time)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/raw/master/weightsGenerate.png)

## Resulte
<br />
### Generation

![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/raw/master/Generation.png)
 
### Test of HyperNet
#### In order to test the HyperNet, two experiment are designed as following
##### Experiment one : Input two different weights and one noise
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/raw/master/diff_weights_same_z1.png)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/raw/master/diff_weights_same_z2.png)
##### Experiment two : Input two different noise and one weight
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/blob/master/diff_z_same_weight1.png)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-GAN/blob/master/diff_z_same_weight2.png)
<br /><br />
## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
