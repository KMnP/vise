Exploring Visual Engagement Signals for Representation Learning
================================================================

<h4>
Menglin Jia, Zuxuan Wu, Austin Reiter, Claire Cardie, Serge Belongie and Ser-Nam Lim
</br>
<span style="font-size: 14pt; color: #555555">
Cornell University, Facebook AI
</span>
</h4>
<hr>

**arXiv**: https://arxiv.org/abs/2104.07767

<div align="center">
  <img width="70%" alt="common supervisory signals" src="images/teaser.png">
</div>

<div align="center">
  <img width="70%" alt="VisE as supervisory signals." src="images/teaser_small.gif">
</div>



**VisE** is a pretraining approach which leverages **Vis**ual **E**ngagement clues as supervisory signals. Given the same image, visual engagement provide semantically and contextually richer information than conventional recognition and captioning tasks.  VisE transfers well to subjective downstream computer vision tasks like emotion recognition or political bias classification.



## 💬 Loading pretrained models

:exclamation:**NOTE**: This is a torchvision-like model (all the layers before the last global average-pooling layer.). Given a batch of image tensors with size ``(B, 3, 224, 224)``, the provided models produce spatial image features of shape ``(B, 2048, 7, 7)``, where ``B``  is the batch size.

#### 

### Loading models with torch.hub

Get the pretrained ResNet-50 models from VisE *in one line*!  

**VisE-250M (ResNet-50)**: this model is pretrained with 250 million public image posts.

```python
import torch
model = torch.hub.load("KMnP/vise", "resnet50_250m", pretrained=True)
```

**VisE-1.2M (ResNet-50)**: This model is pretrained with 1.23 million public image posts.

```python
import torch
model = torch.hub.load("KMnP/vise", "resnet50_1m", pretrained=True)
```



### Loading models manually

|           | Arch      | Size    | Model                                                        |
| --------- | --------- | ------- | ------------------------------------------------------------ |
| VisE-250M | ResNet-50 | 94.3 MB | [download](https://obj.umiacs.umd.edu/publicmodels/vise_250m_rn50_ep10.pth) |
| VisE-1.2M | ResNet-50 | 94.3 MB | [download](https://obj.umiacs.umd.edu/publicmodels/vise_1m_rn50_ep100.pth) |

If you encounter any issues with ``torch.hub``, alternatively you can download the model checkpoints manually, and then following the script below.

```python
import torch
import torchvision

# Create a torchvision resnet50 with randomly initialized weights.
model = torchvision.models.resnet50(pretrained=False)

# Get the model before the global aver-pooling layer.
model = torch.nn.Sequential(*list(model.children())[:-2])

# load the pretrained model from a local path: <CHECKPOINT_PATH>:
model.load_state_dict(torch.load(CHECKPOINT_PATH))
```





## 💬 Citing VisE

If you find VisE useful in your research, please cite the following publication.

```
@misc{jia2021vise,
      title={Exploring Visual Engagement Signals for Representation Learning}, 
      author={Menglin Jia and Zuxuan Wu and Austin Reiter and Claire Cardie and Serge Belongie and Ser-Nam Lim},
      year={2021},
      eprint={2104.07767},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## 💬 Acknowledgments

We thank Marseille who was featured in the teaser photo. 



## 💬 License

VisE models are released under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/KMnP/vise/blob/master/LICENSE) for additional details.