# kaggle-Bengali

## Test

Run the following command to test:

```
$ python test.py
```

## TODO

* Models
    - [ ] DenseNet
    - [ ] EfficientNet
    - [ ] Se-ResNext
    - [ ] support different input size
* Training framework
    - [x] run in CPU
    - [ ] run in GPU
* Optimizers
    - [x] RAdam
    - [ ] Look-ahead
* Loss functions
    - [x] TopkCrossEntropyLoss
    - [x] Weighted multi-task cross entropy
* Cross Validation
    - [ ] 5 folds (need to check the distribution of classes)
* Adversarial Training
    - [ ] optional
* Image Augmentations
    - [ ] cutmix
    - [ ] mixup
    - [ ] cutmix + mixup
* Data
    - [ ] different input size
    - [ ] [progressive resizing](https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20) (optional)
    - [ ] color reverse (black to white, white to black)
    - [ ] fake data
        - [ ] add random image with uniform labels
        - [ ]add digital, english, japanese, even mandarin letters with uniform labels
        - [ ] generate fake Bengali graphemes consist grapheme root, vowel diacritics, and consonant diacritics selected randomly
    - [ ] add [Ekush dataset](https://www.kaggle.com/shahariar/ekush)
    - [ ] removing confusing images
* Test-time augmentation (TTA)
    - [ ] done
* Ensemble
    - [ ] done
* Few-shot learning
    -[ ] [Few-shot learning](https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57896) ()optional
* Metric Learning (?)