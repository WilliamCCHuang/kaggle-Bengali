# kaggle-Bengali

## Test

Run the following command to test our training framework:

```
$ python test.py
```

Once the training is completed, see the `lightning_logs` to check losses do get smaller

```
$ tensorboard --logdir=lightning_logs/version_{x}
```

, where `{x}` is a number and indicates the latest version.

## TODO

* Models
    - [x] DenseNet
    - [x] EfficientNet
    - [x] Se-ResNet
    - [x] Se-ResNext
    - [x] support different input size
* Training framework
    - [x] run in CPU
    - [x] run in GPU
* Optimizers
    - [x] Adam
    - [x] RAdam (reported the same with Adam)
    - [ ] Look-ahead
* Scheduler
    - [ ] ReduceOnPlateau (reported better than OneCycle in discussion)
    - [ ] OneCycle
* Loss functions
    - [x] Top k cross entropy loss
    - [x] Multi-task cross entropy loss (reported better than the others in discussion)
    - [x] Label smoothing cross entropy loss
    - [x] Multi-task label smoothing cross entropy loss
    - [ ] Focal loss
* Cross Validation
    - [ ] 5 folds (need to check the distribution of classes)
* Adversarial Training
    - [ ] optional
* Image Augmentations
    - [ ] cutout
    - [ ] mixup
    - [ ] cutmix
    - [ ] augmix
    - [ ] cutmix + mixup
* Data
    - [ ] different input size
    - [ ] [progressive resizing](https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20) (*optional*)
    - [ ] color reverse (black to white, white to black)
    - [ ] fake data
        - [ ] add random image with uniform labels
        - [ ] add digital, english, japanese, even mandarin letters with uniform labels
        - [ ] generate fake Bengali graphemes consist grapheme root, vowel diacritics, and consonant diacritics selected randomly
    - [ ] add [Ekush dataset](https://www.kaggle.com/shahariar/ekush)
    - [ ] removing confusing images
* Test-time augmentation (TTA)
    - [ ] done
* Ensemble
    - [ ] done
* Few-shot learning
    - [ ] [few-shot learning](https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57896) (*optional*)
* Pseudo label
    - [ ] Noise student
* Metric Learning (?)

## Key points
* grapheme recall dominates