**<span style='font-size:26px;'>CLIP课设目录</span>**

---

[TOC]



## 摘要

CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的一种文本图像对比学习的多模态预训练模型。它的主体结构有两部分，Text Encoder和Image Encoder，分别用于提取文本特征和图像特征，其中Text Encoder采用了Transformer，Image Encoder采用了Vision-Transformer/ResNet，集中了目前深度学习影响力较大的两类神经网络。作者使用400M图像文本对进行预训练，使模型能够得到最大的余弦相似度，并在文章中突出描述了Prompt概念。模型具有强大的Zero-shot能力，在多个数据集上表现出色，如StanfordCars、Food101等，但是在一些细粒度要求高的数据集上表现较差，如EuroSAT，Flowers102等；同时Linear-probe的能力也很强大，在选取的数据集中绝大多数都优于EfficientNet L2 NS。

在论文中，作者首先对NLP中预训练模型的发展做了回顾，介绍了提出CLIP的驱动性因素，即目前的模型大多是在预先设定的数据集上进行训练，如ImageNet等，这给模型泛化性带来问题。之后对CLIP的结构、模型参数、数据集、训练过程等进行了描述。在实验部分，作者测试了CLIP在Zero-shot Transfer、Representation Learning、Distribution Shift等情景下的性能，并与人类表现进行对比。最后，作者对CLIP的表现进行分析，提出后续工作和应用前景，并得出结论。

在本次任务中，我们首先使用单张图片对CLIP进行测试，了解了CLIP的原理和性能；之后在CIFAR数据集上进行了测试，得到了Top1和Top5的分类准确率，符合预期；最后对CLIP带来的下游任务进行了调研，并尝试了部分应用。我们得出结论，CLIP是深度学习理论、硬件设备以及数据规模等条件都发展到一定阶段的产物，集中体现了深度学习领域的发展状况。尽管CLIP在一些细粒度图像分类任务上还有待提高，但其强大的能力为下游任务开辟了新道路，是一个开创性工作。

CLIP等大型预训练模型的出现和它们优秀的性能，让我们思考计算资源聚集后可能会带来的垄断问题。

由于CLIP涉及到众多深度学习领域的概念和工作，我们尚未完全理解，所以报告中可能存在一些错误，敬请指正。





 <div style="page-break-after: always;"></div>

## 一、CLIP介绍

![CLIP](E:\OneDrive\科研论文\CLIP\CLIP\assets\CLIP.png) 

**1. Zero-shot**

CLIP完成Zero-shot任务的过程如上图所示。CLIP主要包括两个部分，Text Encoder和Image Encoder，前者采用了Transformer，后者采用Vision-Transformer（ViT）或ResNet。在预训练时，图像文本对通过Encoder后计算余弦相似度，得到$N$个正样本（如上图（1）所示，位于对角线）和$N^2-N$个负样本。通过训练使正样本余弦相似度达到最大值，达到图像文本匹配。在进行Zero-shot时，首先完成Prompt工作，将Label转化为短语，即进行文本提示，之后再送入Text Encoder，与Image Encoder的结果进行点积，经过sotfmax后即可得出结果。



**2. Prompt** 

采用Prompt是CLIP的一个鲜明特征。Prompt放开了分类的范围，摆脱数据集基础类的限制，可以从自然语言中提取标签，只需要对比图片中是否含有文本所描述（感兴趣）的对象；加入了语义理解，使分类的范围变得多重，比如同时对物体类别、颜色等进行分类（a green toy）；也避免了只有一个单词带来的歧义，文中给出例子Construction cranes and cranes（吊车和鹤）。

作者在进行图像分类测试中，在每个数据集上分别使用了各自的Classes and prompts，其中Prompts使用的大多为photo、view、type等词汇再加上一些修饰。附录1给出了CIFAR10的例子。官方也给出了ImageNet的示例，采用了1000个类别和80个Templates的组合。



**3. Linear Probe**

作者使用Linear Probe的方法在多个数据集上进行图像分类测试，即冻结网络只对最后FC层分类头进行训练。在此没有使用Fine tune，一是为了测试预训练模型的性能，二是由于模型庞大，Fine tune带来的变化也大，可能使模型变差。



**4. Models**

这一部分较难理解，细节众多，最需要理论积淀和深入研究。

Text Encoder部分是在已有的Transformer模型上针对CLIP进行结构微调，有63M参数、12层、512位宽、8个Attention heads，文本长度上限为76。Image Encoder部分，作者训练了8个模型，ResNet-50、ResNet-101、4x/16x/64x-ResNet-50、ViT-B/32、ViT-B/16、ViT-L/14。实际采用了ViT-L/14@336px，是在ViT-L/14基础上在336像素分辨率又训练了一个epoch。

训练参数：32 epochs、a large minibatch size of 32768、Adam Optimizer。

超参数设置：在训练了一个epoch的ResNet-50基础上，混合使用grid searches，random search，manual tuning。

训练时长：RN50x64 - 18 days on 592 V100 GPUs  ,  ViT-L/14 - 12 days on 256 V100 GPUs 。

其他细节： temperature parameter τ=0.07，Mixed-precision(16/32bit mixing to save memory)，gradient checkpointing，half-precision Adam statistics， half-precision stochastically rounded text encoder weights。这一部分没有深究，但提供一些训练技巧，所以在此列出。





## 二、单图测试

我们一共在6张图片上做了测试，可以看出CLIP在偏宏观的类别和描述上表现较好，但涉及到细节表现不佳。据此也可以推断出CLIP在一些类间差异较小的数据集上性能较差。见附件

下图第一行到第二行、从左至右依次为1-6号图片，选取了不同类型的图片。

![sample](E:\OneDrive\科研论文\CLIP\CLIP\assets\sample.jpg) 

1. 图片1在文本部分依次增加’map‘，’a white map‘，’a colorful map‘，’Chinese province‘，输出概率不断上升，在具体到某一省时也会使’Chinese Province‘的概率下降，但仍为最高。
2. 图片2可以识别'a person with a dog'，'blue'，但无法识别'a person laughing with a dog'。加入'anime'后，'anime'明显占优。
3. 图片3是为了与图片2进行对比而从google搜索得到的一张'a person laughing with a dog'，该标签在Zero-shot结果中占绝对优势。图片2和3的对比表明训练数据本身带有一定bias，比如主要是人们日常见到的图片，对于一些不常见（out of distribution）的图片也能获取大的特征但细节判别较差。
4. 图片4为北航校徽。'emblem of beihang university'得到最大概率，最初测试时'beihang'一个单词就可以占据绝对优势。
5. 图片5为NLP一些模型的对比，图中没有一个地方写NLP，但输出结果中'nlp'概率最大，且高于'GPT','BERT'等，表明在获取宏观语义上能力较好。
6. 图片6为一张创意合成图片。'architecture'，'sky'，'sky and architecture'概率依次上升，但'green grass'，'lawn'等概率很小，完全未识别到草坪，考虑可能的原因是图片中的草坪不是真草坪，类似图2和3的差别，再次表明CLIP在非现实数据上表现也较差。





## 三、CIFAR数据集表现

我们分别在CIFAR10、CIFAR100数据集上进行了Zero shot和 Linear Probe的测试，符合预期，即类间差异变小的情况下，CLIP性能下降。实验环境为Google Colab。代码见附件，结果如下。

**1. Zero shot**

correct和precision为Top1正确率，带后缀5为Top5正确率，使用的模型为ViT-B/32，Prompt采用了“a photo of {class}"。

![image-20221117011328712](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117011328712.png) 

![image-20221117011334795](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117011334795.png) 

**2. Linear Probe**

CIFAR10：![image-20221117011741126](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117011741126.png) 

CIFAR100：![image-20221117011715222](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117011715222.png) 





## 四、CLIP性能分析

论文中给出了详细的对比分析，集中于CLIP的Zero-shot和Linear-Probe性能上。这一部分涉及到诸多模型，并且在过去一段时间各种模型又有了进一步发展，由于我们对这些模型没有做到完全了解，所以在此仅列出一些较为直观的分析，希望能透过这些主要的分析来探视到CLIP的特点。

**1. Zero-shot CLIP对比ResNet-50**

可以看出CLIP在一些语义强烈，类间区别较大的数据集上表现良好；而在细化、抽象的数据集上（如DTD，EuroSAT等）表现较差。其中，MNIST数据集上CLIP的性能明显更差，结合单图测试的结果，初步说明CLIP在处理合成图像而非自然图像时劣势会更明显。

<img src="E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117025127311.png" alt="image-20221117025127311" style="zoom:50%;" /> 



**2. CLIP对比Few-shot linear probes**

Zero-shot CLIP已经可以做到较高准确度。在给出少量训练样本时CLIP性能反而有下降，考虑训练样本使网络参数变化较大而带有个体特征，降低了泛化性，而随着训练样本增多泛化性又逐渐回归并对特定类增强，所以在最后LP CLIP性能又超出了Zero-shot CLIP。

<img src="E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117025111952.png" alt="image-20221117025111952" style="zoom:50%;" /> 

**3. Zero-shot CLIP对比Few-shot CLIP**

Zero-shot CLIP在整体上低10-25个百分比。与分析2相同，当CLIP针对特定任务进行训练并且数据量足够时性能会提升。

<img src="E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117025048987.png" alt="image-20221117025048987" style="zoom:50%;" /> 

**4. Linear Probe CLIP对比SOTA CV models 表征学习**

Linear Probe CLIP 平均表现更好，并且ViT-CLIP优于ResNet-CLIP。

<img src="E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117025029469.png" alt="image-20221117025029469" style="zoom:50%;" /> 

**5. Zero-shot CLIP应对Distribution shift**

Zero-shot CLIP明显好于ResNet101(ImageNet)。一个可能的问题是作者选择了特定的数据（’banana’），并且尽管选择的数据存在Distribution shift，但只是有一些是画作或艺术作，整体上仍然是自然存在的，而不是合成的，所以CLIP表现良好。

<img src="E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117024958675.png" alt="image-20221117024958675" style="zoom:50%;" /> 

**6. CLIP与人类对比**

由于人类对类别熟悉程度可能较低，所以CLIP明显占优。当提前给人类数据时，人类的表现会提高，而CLIP会出现下降。可能的原因与分析2的Few-shot对比相同。

![image-20221117024539059](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221117024539059.png)



## 五、StyleCLIP

CLIP有诸多应用，如StyleCLIP、CLIPDraw、StyleCLIPDraw图像生成，物体检测与分割，视频检索等。这些应用整体上都利用了CLIP对于自然物体宏观特征的辨识能力。

在CLIP诞生后，各种基于CLIP的工作陆续发表，我们主要关注了StyleCLIP，即StyleGAN与CLIP的组合。StyleCLIP可以做到通过语言描述来对原始图像进行修饰并生成新的图像。下图为一个例子，采用的语言描述为"A woman with purple hair"，可以看出经过处理后生成了对应的图片，基本完成了任务。但是从生成的图中也可以看出StyleCLIP仍然有一些局限性，比如嘴唇部分出现了明显变化，背景部分也变为了紫色，右下角衣服装饰变化，说明StyleCLIP在细节上做的不够好。由于CLIP实际上是作为一个预训练模型参与其中的，并且尽管CLIP使用了巨大的数据集，但是可能更关注于模型泛化而在训练时扩大类别，对于某一具体类别的图片数量可能比较少，因此实际上仍然受限于训练时的数据特征，无法完全将背景、嘴唇、头发等语义特征完全解耦。另一方面，GAN也存在局限性。以上现象被称为特征纠缠。

![image-20221118142105693](E:\OneDrive\科研论文\CLIP\CLIP\assets\image-20221118142105693.png) 

另外，附件中给出了一个StyleCLIP生成的一段变化视频，这个视频展示了使用StyleCLIP按照'a really sad person'进行处理的过程。最终结果并不够好，图片整体上仍然是笑容为主。观察整个变化过程，可以看出额头出现皱眉等特征，而嘴唇部分处理较少。一方面，模型可能并没有理解到需要改变嘴部特征，另一方面由于“笑”这一特征非常明显，要将这一特征改变，需要从整体上破坏构图，对嘴唇部分进行强行处理，还是有一定难度的。





## 六、总结

1. CLIP在Zero shot, 使用Linear Probe进行图像分类等方面都有强大的能力，为下游任务开辟了道路。
2. 在数据集选择上自带bias，作者也在文章中提到实验参数等的选择受ImageNet的影响，同时模型性能也受下游任务影响。
3. 它是深度学习理论、硬件设备以及数据规模等条件都发展到一定阶段的产物，集中体现了深度学习领域的发展状况。
4. 尽管CLIP在一些细化、抽象的图像任务上还有待提高，但其作为一个开创性工作值得深入研究发展。
5. CLIP推出已经有一段时间，在此期间多模态预训练模型也取得了诸多进展，并且模型泛化性进一步提升。一个优秀的预训练模型在经过针对性调整后完成各种下游任务，这是深度学习的前景所在。也许未来会出现一段时间内某个预训练模型赢者通吃，再通过不断迭代预训练模型，达到垄断的效果，这更加突出了计算资源聚集的重要性。未来深度学习会否形成数据与计算资源密集型产业，值得我们深思。



<div style="page-break-after:always"></div>

## 参考文献

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh : Learning Transferable Visual Models From Natural Language Supervision. Arxiv Preprint: https://arxiv.org/abs/2103.00020

[2] Or Patashnik, Zongze Wu, Eli Shechtman : StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. Arxiv Preprint: https://doi.org/10.48550/arXiv.2103.17249

[3] Kevin Frans, L.B. Soros, Olaf Witkowski : CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders. Arxiv Preprint: https://doi.org/10.48550/arXiv.2106.14843

[4] Peter Schaldenbrand, Zhixuan Liu, Jean Oh : StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Synthesis . Arxiv Preprint: https://doi.org/10.48550/arXiv.2111.03133

[5] https://openai.com/blog/clip/

[6] https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.337.search-card.all.click

[7] https://zhuanlan.zhihu.com/p/412126626

[8] https://zhuanlan.zhihu.com/p/526449242

[9] https://zhuanlan.zhihu.com/p/63230738

[10] https://github.com/openai/CLIP

[11] https://github.com/ndb796/StyleCLIP-Tutorial









<div style="page-break-after:always"></div>

## 附录1

#### CIFAR10 Classes and Templates

```python
## CIFAR10 
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]
```

