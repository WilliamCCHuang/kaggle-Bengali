import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from image_utils import resize, image_to_tensor


class GradCAM:

    def __init__(self, model, idx2label=None):
        """
        Grad-CAM
        
        Arguments:
            model {nn.Module} -- A PyTorch model
            idx2label {function} -- A dictionary that can map a class index to a class label, \
                otherwise the class index would be shown instead. 
        """

        assert 'input_size' in dir(model), '`model` should have an attribute called `input_size`.'
        assert 'features' in dir(model), '`model` should have a method called `features`.'
        assert 'logits' in dir(model), '`model` should have a method called `logits`.'

        self.model = model
        self.outputs = None
        self.gradients = None
        self.predictions = None
        self.feature_maps = None
        self.idx2label = idx2label
        self.input_size = self.model.input_size[1:]  # (H, W)
        
        self.model.eval()
    
    def _save_gradients(self, grad):
        self.gradients = grad.detach().numpy()[0]
    
    def _forward_pass(self, img, use_logits):
        feature_maps = self.model.features(img)
        logits = self.model.logits(feature_maps)
        probas = F.softmax(logits, dim=-1)
        
        feature_maps.register_hook(self._save_gradients)
        
        if use_logits:
            self.outputs = logits
        else:
            self.outputs = probas
        
        self.predictions = probas.detach().numpy().squeeze()  # (n_classes,)
        self.feature_maps = feature_maps.detach().numpy().squeeze()  # (C, H, W)
        
    def _backward_pass(self, class_idx):
        onehot_target = torch.zeros_like(self.predictions, dtype=torch.float)  # (n_classes,)
        onehot_target = torch.unsqueeze(onehot_target, dim=0)  # (1, n_classes)
        onehot_target[0, class_idx] = 1.0
        
        self.model.zero_grad()
        self.outputs.backward(gradient=onehot_target)
        
    def _get_class_idx(self, top_i):
        class_idx = self.predictions.argsort()
        class_idx = class_idx[-top_i]
        
        return class_idx
    
    def generate_heatmap(self, img, class_idx, counterfactual=False, relu_on_gradients=False,
                         use_logits=True):
        if isinstance(img, Image.Image):
            img = image_to_tensor(img)
        
        self._forward_pass(img, use_logits)
        self._backward_pass(class_idx)
        
        if relu_on_gradients:
            weights = np.mean(np.maximum(self.gradients, 0), axis=(1, 2))
        else:
            weights = np.mean(self.gradients, axis=(1, 2))
        weights = weights.reshape((-1, 1, 1))
        
        if counterfactual:
            weights = - weights
        
        heatmap = np.sum(weights * self.feature_maps, axis=0) # weighted sum over feature maps
        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize between 0-1
        heatmap = np.uint8(heatmap * 255)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(self.input_size, Image.ANTIALIAS)
        heatmap = np.array(heatmap)
        
        return heatmap
        
    def plot_image_heatmap(self, img, top=1, counterfactual=False, relu_on_gradients=False,
                           use_logits=True, save_path=None):
        img = resize(img, size=self.input_size)  # (H, W, C)

        if isinstance(img, Image.Image):
            img = image_to_tensor(img)  # (1, C, H, W)

        plt.imshow(img)
        plt.axis('off')
        
        cols = top + 1
        plt.figure(figsize=(4 * cols, 4))
        for i in range(cols):
            if i == 0:
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.title('original image')
                plt.axis('off')
            else:
                heatmap = self.generate_heatmap(img_tensor, class_idx, counterfactual, relu_on_gradients, use_logits)

                class_idx = self._get_class_idx(top_i=i)
                proba = self.predictions[class_idx]
                label = self.idx2label[class_idx] if self.idx2label else str(class_idx)
                
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.imshow(heatmap, cmap='rainbow', alpha=0.7)
                plt.title('{} ({:.3f})'.format(label, proba))
                plt.axis('off')
                
        if save_path:
            plt.savefig(save_path)
            