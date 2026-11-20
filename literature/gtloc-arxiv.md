Title:

Content selection saved. Describe the issue below:

Description:

![](https://arxiv.org/static/base/1.0.1/images/icons/smileybones-small.svg)arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2507.10473v2 \[cs.CV\] 25 Jul 2025

# GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space

David G. Shatwell
Affiliation: Center for Research in Computer Vision, University of Central Florida
Ishan Rajendrakumar Dave
Affiliation: Adobedavid.shatwell@ucf.edu  idave@adobe.com  swetha.sirnam@ucf.edu  shah@crcv.ucf.eduSirnam Swetha
Affiliation: Center for Research in Computer Vision, University of Central Florida
Mubarak Shah
Affiliation: Center for Research in Computer Vision, University of Central Florida

###### Abstract

Timestamp prediction aims to determine when an image was captured using only visual information, supporting applications such as metadata correction, retrieval, and digital forensics. In outdoor scenarios, hourly estimates rely on cues like brightness, hue, and shadow positioning, while seasonal changes and weather inform date estimation. However, these visual cues significantly depend on geographic context, closely linking timestamp prediction to geo-localization. To address this interdependence, we introduce GT-Loc, a novel retrieval-based method that jointly predicts the capture time (hour and month) and geo-location (GPS coordinates) of an image. Our approach employs separate encoders for images, time, and location, aligning their embeddings within a shared high-dimensional feature space. Recognizing the cyclical nature of time, instead of conventional contrastive learning with hard positives and negatives, we propose a temporal metric-learning objective providing soft targets by modeling pairwise time differences over a cyclical toroidal surface. We present new benchmarks demonstrating that our joint optimization surpasses previous time prediction methods, even those using the ground-truth geo-location as an input during inference. Additionally, our approach achieves competitive results on standard geo-localization tasks, and the unified embedding space facilitates compositional and text-based image retrieval.

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2507.10473v2/teaser_cropped.png)Figure 1: GT-Loc: Our Unified Approach vs. Prior Methods. By mapping image, location and time into a single multimodal embedding space, our method can be used for (a) simultaneous image-to-location and image-to-time retrieval, (b) composed geotemporal-to-image retrieval. In contrast, current methods are limited to only (c) location \[ [5](https://arxiv.org/html/2507.10473v2#bib.bib5 ""), [3](https://arxiv.org/html/2507.10473v2#bib.bib3 "")\], (d) time \[ [27](https://arxiv.org/html/2507.10473v2#bib.bib27 "")\] or (e) geo-temporal classification \[ [45](https://arxiv.org/html/2507.10473v2#bib.bib45 "")\] or (f) image-to-GPS retrieval \[ [11](https://arxiv.org/html/2507.10473v2#bib.bib11 ""), [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\].

Estimating the capture time and geo-location of images is crucial for applications ranging from digital forensics to ecological studies and social media management. In digital forensics, accurate timestamps verify image authenticity and help detect manipulation, particularly when camera calibrations are suspect. This capability is essential for reconstructing events from timestamped images during accidents or natural disasters, providing critical information to first responders. Ecological studies benefit from time-ordered images to monitor changes in landscapes and wildlife, while precise timestamps in social media enhance content management and chronological sorting.

Despite its importance, predicting time from images presents several challenges because of the intricate relationship between temporal cues and location-specific factors. Time-of-day (ToD; i.e., hours) and time-of-year (ToY; i.e., months) manifest differently in images due to variables like scene brightness, shadows, weather, and seasonal changes, making it difficult to establish consistent patterns. The complexity of the task is further compounded as the visual appearance of specific hours varies substantially across different months and locations, influenced by the amount and relative exposure to sunlight. Additionally, the representation of months fluctuates across various latitudes, with regions near the equator experiencing relatively stable climate conditions year-round compared to regions at higher latitudes.

Most existing methods \[ [45](https://arxiv.org/html/2507.10473v2#bib.bib45 ""), [27](https://arxiv.org/html/2507.10473v2#bib.bib27 ""), [22](https://arxiv.org/html/2507.10473v2#bib.bib22 "")\] rely heavily on GPS metadata for accurate time estimation, while state-of-the-art geo-localization models (e.g., PIGEON \[ [5](https://arxiv.org/html/2507.10473v2#bib.bib5 "")\], GeoCLIP \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\]) excel at coarse location prediction. Yet predicting both time and location simultaneously without supplementary inputs remains unsolved, and few works tackle the complementary task of image retrieval (Figure [1](https://arxiv.org/html/2507.10473v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")). This challenge is further exacerbated by the absence of standardized datasets and evaluation protocols: many approaches use custom or poorly documented splits, hindering fair comparisons and consistent benchmarking across studies.

In this paper, we introduce GT-Loc, a retrieval-based approach for joint time prediction and geo-localization. We conceptualize time prediction as a retrieval problem, representing time as a month-hour pair. A schematic diagram of our framework is shown in Figure [2](https://arxiv.org/html/2507.10473v2#S3.F2 "Figure 2 ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). Building upon the CLIP-initialized visual model, our goal is to learn a shared embedding space where we can align visual (image), time, and location modalities. For time prediction, we propose a novel time representation that considers the cyclical nature of months and hours over a toroidal manifold. We then project these representation into a multi-scale, high-dimensional time embedding using random Fourier features (RFFs) \[ [33](https://arxiv.org/html/2507.10473v2#bib.bib33 "")\]. Next, to learn the alignment between the time and image embeddings, we explore several possibilities. Existing contrastive learning methods, including CLIP \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\] and SimCLR \[ [2](https://arxiv.org/html/2507.10473v2#bib.bib2 "")\], use other batch instances as negative samples. Such a strategy succeeds in image-location losses used by Contrastive Spatial Pre-Training (CSP) \[ [18](https://arxiv.org/html/2507.10473v2#bib.bib18 "")\], GeoCLIP \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\], and SatCLIP \[ [11](https://arxiv.org/html/2507.10473v2#bib.bib11 "")\], due to the significant variation of visual appearance with respect to geographical location. In contrast, image-time alignment suffers because temporal neighbors often look nearly identical: hours and months blend smoothly in appearance, so designating adjacent time points as negatives undermines effective alignment. Instead of defining positive-negative pairs as in contrastive learning, we propose a novel Temporal Metric Learning approach, which encourages similarity between two instances based on their time difference. To build the target metric for our proposed loss, we use the toroidal distance between the times of each instance pair to consider the cyclic nature of time. This approach enhances performance without the need for explicit assignment of positive and negative samples, providing a more effective and efficient solution for time prediction.

By mapping the image, location, and time modalities into a unified feature space, our model is able to perform compositional retrieval tasks. For instance, given a specific time and location, it can efficiently retrieve all corresponding images from a gallery that closely match the specified criteria.

In summary, our main contributions are the following:

- •


A framework for joint time-of-capture prediction and geo-localization by aligning the image, time and location embeddings in a shared multimodal feature space using contrastive learning.

- •


The first retrieval-based method for time-of-capture prediction, where we propose a novel time representation as normalized month-hour pairs, considering its cyclic nature.

- •


A novel Temporal Metric Learning (TML) loss function for image-time alignment with soft targets, eliminating the need to assign positive and negative samples to the anchor. Since both hours and months are cyclic, we employ a toroidal distance instead of a regular ℓ2\\ell\_{2} distance which results in improved performance.

- •


A new standard benchmarks for time prediction, demonstrating that our jointly optimized time-location method surpasses time-only optimized baselines and competes well with expert geo-localization methods. Our shared embedding space further facilitates downstream tasks like compositional and text-based retrieval.


## 2 Related Work

Time-of-capture prediction: Time-of-capture prediction is a relatively new problem that has only been directly addressed by a handful of prior works. [Tsai et al. \[36\]](https://arxiv.org/html/2507.10473v2#bib.bib36 "") proposed a physically inspired method to infer the time of day by estimating the Sun’s position and camera orientation, but their approach requires sky visibility and additional metadata, such as GPS coordinates and access to an external image database. In a different line of research, [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "") introduced a data-centric approach to learn geo-temporal image features. Their model uses an image, location, and time encoders to generate mid-level features, which are subsequently passed to a set of classifiers for predicting time and location as discrete classes. However, their evaluation shows that providing the location as an input is crucial for predicting the time of day with reasonable accuracy. Similarly, [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 "") proposed a hierarchical model to predict the month, hour, and week of capture, but this method also assumes known geo-location, limiting its real-world applicability. In contrast, our model relies solely on images to generate accurate time predictions using a retrieval approach in a continuous shared feature space, with resolution determined by a gallery of arbitrary size rather than discrete classes.

Other works have also explored the time-of-capture task indirectly. [Li et al. \[16\]](https://arxiv.org/html/2507.10473v2#bib.bib16 "") presented an algorithm to verify image capture time and location by comparing the sun position, computed from the claimed time and location, with the actual sun position derived from shadow length and orientation. However, their approach assumes that latitude, time-of-day, and time-of-year are given, with only one potentially corrupted. [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "") proposed a model for time-of-capture verification using a data-centric approach, involving four encoders for ground-level images, timestamps, geo-locations, and satellite images, fed into a binary classifier to predict time consistency. Similarly, [Salem et al. \[26\]](https://arxiv.org/html/2507.10473v2#bib.bib26 "") proposed a model to generate a global-scale dynamic map of visual appearance by matching visual attributes across images annotated with timestamps and GPS coordinates. Both [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "") and [Salem et al. \[26\]](https://arxiv.org/html/2507.10473v2#bib.bib26 "") show qualitative results on time prediction, but are limited by their dependency on geo-location. In contrast, our method accurately estimates both GPS coordinates and timestamps using only images.

Several additional methods explore problems adjacent to time prediction. [Jacobs et al. \[8\]](https://arxiv.org/html/2507.10473v2#bib.bib8 "") proposed an algorithm for geo-locating static cameras by comparing temporal principal components of yearly image sequences with those from a gallery of known locations. For shadow detection, [Lalonde et al. \[14\]](https://arxiv.org/html/2507.10473v2#bib.bib14 "") used a multi-stage method to extract pixel-wise ground-shadow features and find edges using CRF optimization, while [Wehrwein et al. \[40\]](https://arxiv.org/html/2507.10473v2#bib.bib40 "") used illumination ratios to label shadow points in a 3D reconstruction and compute dense shadow labels in pixel space. Another series of works estimate the sun position for various downstream applications, such as computing camera parameters from time-lapses \[ [13](https://arxiv.org/html/2507.10473v2#bib.bib13 "")\] and determining outdoor illumination conditions \[ [14](https://arxiv.org/html/2507.10473v2#bib.bib14 ""), [6](https://arxiv.org/html/2507.10473v2#bib.bib6 "")\] from single images. Adapting these methods for time prediction would require additional metadata, which might not be available during inference. For example, even with correct sun position prediction, day of the year, geo-location, and compass orientation are needed to accurately predict the hour.

Although the above works contribute to time prediction, they either require additional input metadata like GPS coordinates, or are not reliable in the absence of specific temporal cues. In contrast, our proposed GT-Loc model aims to predict both ToY and ToD from a single image without relying on any additional metadata, making it more broadly applicable for real-time prediction tasks.

Global geo-localization: Geo-localization, the task of estimating the geographic coordinates of an image, has gained substantial popularity in recent years. Traditionally, geo-localization methods have adopted either a classification approach \[ [41](https://arxiv.org/html/2507.10473v2#bib.bib41 ""), [28](https://arxiv.org/html/2507.10473v2#bib.bib28 ""), [38](https://arxiv.org/html/2507.10473v2#bib.bib38 ""), [20](https://arxiv.org/html/2507.10473v2#bib.bib20 ""), [23](https://arxiv.org/html/2507.10473v2#bib.bib23 ""), [12](https://arxiv.org/html/2507.10473v2#bib.bib12 "")\] or an image retrieval approach \[ [25](https://arxiv.org/html/2507.10473v2#bib.bib25 ""), [29](https://arxiv.org/html/2507.10473v2#bib.bib29 ""), [30](https://arxiv.org/html/2507.10473v2#bib.bib30 ""), [35](https://arxiv.org/html/2507.10473v2#bib.bib35 ""), [49](https://arxiv.org/html/2507.10473v2#bib.bib49 ""), [50](https://arxiv.org/html/2507.10473v2#bib.bib50 "")\]. The classification approach divides the Earth into a fixed number of geo-cells or uses the city label, assigning the center coordinate of the selected class as the GPS prediction. However, this can result in significant errors depending on the size of the geo-cells, even when the correct class is selected. In contrast, the image retrieval approach compares a query image to a gallery and retrieves the image with the highest similarity. GeoCLIP \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\] addresses the limitations of traditional approaches by framing global geo-localization as a GPS retrieval problem. It leverages the pretrained CLIP \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\] ViT and employs contrastive learning to align image and location embeddings in a shared feature space. Other methods, such as PIGEON \[ [5](https://arxiv.org/html/2507.10473v2#bib.bib5 "")\], use a hybrid strategy: first using image classification to identify the top-kk geo-cells with the highest probability, followed by a secondary retrieval stage for refinement within and across geo-cells. However, PIGEON’s dependence on additional metadata for training—such as administrative boundaries, climate, and traffic—poses a significant limitation. Finally, recent methods such as Img2Loc \[ [47](https://arxiv.org/html/2507.10473v2#bib.bib47 "")\] have begun leveraging multimodal large language models (MLLMs) and retrieval-augmented generation (RAG) to achieve competitive geo-localization performance without the need for dedicated training. However, the effectiveness of these methods is highly dependent on the underlying MLLM and results in substantial inference overhead.

Geo-spatial dual-encoder methods:
The success of CLIP has inspired to leverage its architecture for geo-spatial tasks. SatCLIP \[ [11](https://arxiv.org/html/2507.10473v2#bib.bib11 "")\] aligns satellite imagery and natural images in a shared feature space, enabling cross-modal retrieval and localization. In a similar fashion, [Zavras et al. \[43\]](https://arxiv.org/html/2507.10473v2#bib.bib43 "") proposed a method for aligning complementary remote sensing modalities beyond RGB with the CLIP encoders. Other works from [Mai et al. \[18\]](https://arxiv.org/html/2507.10473v2#bib.bib18 "") and [Mac Aodha et al. \[17\]](https://arxiv.org/html/2507.10473v2#bib.bib17 ""), employ a dual image-location encoder architecture to learn robust location representations from images. However, their ultimate goal is not to geo-locate images. Instead, they use the learned embeddings from the image encoder for downstream tasks, such as image classification. These methods demonstrate the effectiveness of dual-encoder architectures for geo-spatial problems, motivating our approach of using a triple encoder architecture for joint time-of-capture and location prediction.

## 3 Method

Given a training dataset St​r​a​i​n={(Ii,Gi,Di)}i=1NS\_{train}=\\{(I\_{i},G\_{i},D\_{i})\\}\_{i=1}^{N} consisting of image IiI\_{i}, GPS coordinates GiG\_{i} and date-time DiD\_{i} triplets, our objective is to train a model that can simultaneously predict the location, time-of-day (ToD) and time-of-year (ToY) from unseen images.
Our GT-Loc method consists of three encoders: Image Encoder (𝒱\\mathcal{V}), Location Encoder (ℒ\\mathscr{L}), and Time Encoder (𝒯\\mathcal{T}) as shown in Figure [2](https://arxiv.org/html/2507.10473v2#S3.F2 "Figure 2 ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). Both geo-localization and time prediction are framed as a retrieval problem. Given a query image IQ∈Se​v​a​lI^{Q}\\in S\_{eval}, we compute an image embedding, VQ=𝒱⁡(IQ)V^{Q}=\\mathcal{V}(I^{Q}), using a pre-trained Vision Transformer. Similarly, given a gallery of latitude-longitude pairs, and a gallery of timestamps, we respectively compute galleries of location embeddings LkQ=ℒ⁡(GkQ)L^{Q}\_{k}=\\mathscr{L}(G^{Q}\_{k}) and time embeddings TkQ=𝒯⁡(DkQ)T^{Q}\_{k}=\\mathcal{T}(D^{Q}\_{k}). In order to predict the location and time, the image embedding is compared against both galleries. The GPS and timestamp with the highest cosine similarity to the image are selected as predictions.

![Refer to caption](https://arxiv.org/html/2507.10473v2/method.png)Figure 2: Overview of GT-Loc: GT-Loc uses an image encoder 𝒱⁡(⋅)\\mathcal{V}(\\cdot), location encoder ℒ⁡(⋅)\\mathscr{L}(\\cdot) and time encoder 𝒯⁡(⋅)\\mathcal{T}(\\cdot) to generate a set of image ViV\_{i}, location LiL\_{i} and time TiT\_{i} embeddings.
Leveraging the CLIP \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\] pretrained ViT-L/14 as image encoder, we aim to align its image embedding to both location and time embeddings.
The image-location alignment is learned through a regular CLIP-like loss \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\] and the image-time alignment is learned through our proposed Temporal Metric Learning.

A fundamental prerequisite for retrieval is the alignment of image, location, and time modalities within a shared multimodal embedding space. To achieve this, our framework is optimized using two multimodal alignment objectives: (1) Image-Location alignment, and (2) Image-Time alignment. Furthermore, to capitalize on large-scale visual pretraining, we employ the pretrained CLIP ViT-L/14 as our image encoder, projecting it into our shared embedding space using a trainable multi-layer perceptron (MLP).

### 3.1 Image-Location Alignment

We adopt GeoCLIP for the image-location modality alignment. Given a latitude-longitude pair GiG\_{i}, it first uses Equal Earth Projection (EEP) to mitigate the distortion of the standard GPS coordinate system and provide a more accurate representation Gi′G\_{i}^{\\prime}. Then, Random Fourier Features (RFF) are used to map the 2D representation into a rich high-dimensional representation at three scales (MM) using projection matrices γ⁡(⋅)\\gamma(\\cdot) with different frequencies σi∈{20,24,28}\\sigma\_{i}\\in\\left\\{2^{0},2^{4},2^{8}\\right\\}. Lastly, the RFFs are passed to a set of MLPs fif\_{i} and added together, forming a single multi-scale feature vector. This can be mathematically expressed as the following equation:

|     |     |     |     |
| --- | --- | --- | --- |
|  | Li=ℒ⁡(Gi)=∑i=1Mfi​(γ⁡(E​E​P​(Gi),σi)).L\_{i}=\\mathscr{L}(G\_{i})=\\sum\_{i=1}^{M}f\_{i}(\\gamma(EEP(G\_{i}),\\sigma\_{i})). |  | (1) |

Next, to compute the image-location contrastive loss ℒl​o​ci\\mathcal{L}\_{loc}^{i}, we consider a set of PP augmented image Vi​jV\_{ij} and location Li​jL\_{ij} embeddings (j∈1,…,Pj\\in{1,\\dots,P}). For the batch with size BB with SS additional location embeddings stored in a continually updated dynamic queue, and temperature τ\\tau, the loss is given by:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒl​o​c(i)=−∑j=1Plog(exp⁡(Vi​j⋅Li​j/τ)∑k=1B+Sexp⁡(Vk​j⋅Lk​j/τ)).\\mathcal{L}\_{loc}^{(i)}=-\\sum\_{j=1}^{P}\\log\\left(\\frac{\\exp(V\_{ij}\\cdot L\_{ij}/\\tau)}{\\sum\_{k=1}^{B+S}\\exp(V\_{kj}\\cdot L\_{kj}/\\tau)}\\right). |  | (2) |

![Refer to caption](https://arxiv.org/html/2507.10473v2/loss.png)Figure 3: (a) Proposed Temporal Metric Learning loss ℒt​i​m​e\\mathcal{L}\_{time}:
We compute the image-time similarity matrix by taking the cosine distance between the image and time embedding of the all instances of the batch. We then obtain the target distance matrix by computing the cyclic toroidal time difference between each pair.
As shown in the red highlighted box, we take the iith row of both matrices and normalize them using the softmax function σ⁡(⋅)\\sigma(\\cdot) and 1−σ⁡(⋅)1-\\sigma(\\cdot) respectively, resulting in two probability mass functions 𝐩i\\mathbf{p}\_{i} and 𝐪i\\mathbf{q}\_{i}. The loss is then given by the cross-entropy (CE) between 𝐩i\\mathbf{p}\_{i} and 𝐪i\\mathbf{q}\_{i}.
(b) Our proposed distance metricδi,j\\delta\_{i,j}: Assume we have two normalized month-hour pairs Di′=(θi,ϕi)D\_{i}^{\\prime}=(\\theta\_{i},\\phi\_{i}) and Dj′=(θj,ϕj)D\_{j}^{\\prime}=(\\theta\_{j},\\phi\_{j}). Since month and hours are periodic, they repeat infinitely in both directions of the θ\\theta-ϕ\\phi plane. Using the ℓ2\\ell\_{2} distance overestimates the real distance between the two times, but our proposed toroidal time distance δi,j\\delta\_{i,j} is able to provide a correct estimate by considering the minimum distance between Di′D\_{i}^{\\prime} and the periodic copies of Dj′D\_{j}^{\\prime}.

### 3.2 Time Representation

The capture time of an image is usually a Unix timestamp, an integer tracking the seconds (or milliseconds) elapsed since January 1, 1970.
We discard the year information from the timestamp, as predicting the year is beyond the scope of this work, and instead focus on Time-of-Year (ToY; i.e., month) and Time-of-Day prediction (ToD; i.e., hour). Both ToY and ToD are cyclical, with periods of 12 months and 24 hours, respectively. To convert the Unix timestamp UiU\_{i} into a time representation that focuses on the months and hours, we transform it into a date-time tuple Di=unix2tuple​(Ui)=(mi,di,Hi,Mi,Si)D\_{i}=\\mathrm{\\texttt{unix2tuple}}(U\_{i})=(m\_{i},d\_{i},H\_{i},M\_{i},S\_{i}) with the month, day, hour, minute, and second. From this tuple, we then compute a new time representation composed of the normalized cyclic month-hour pair Di′=(θi,ϕi)D\_{i}^{\\prime}=(\\theta\_{i},\\phi\_{i}) using

|     |     |     |     |
| --- | --- | --- | --- |
|  | θi=112​((mi−1)+(di−1)𝒟⁡(mi)),\\displaystyle\\theta\_{i}=\\frac{1}{12}\\left((m\_{i}-1)+\\frac{(d\_{i}-1)}{\\mathcal{D}(m\_{i})}\\right), |  | (3) |

|     |     |     |     |
| --- | --- | --- | --- |
|  | ϕi=124​(Hi+Mi60+Si3600),\\displaystyle\\phi\_{i}=\\frac{1}{24}\\left(H\_{i}+\\frac{M\_{i}}{60}+\\frac{S\_{i}}{3600}\\right), |  | (4) |

where 𝒟⁡(mi)\\mathcal{D}(m\_{i}) is the number of days in month mim\_{i}. We represent the sequence of operations to convert a Unix timestamp to normalized cyclic month-hour pair as Di′=unix2cyclic​(Ui)D\_{i}^{\\prime}=\\mathrm{\\texttt{unix2cyclic}}(U\_{i}).

Time encoding:
By representing time as a pair of real-valued numbers, the problem of time prediction becomes similar in nature to geo-localization. Instead of retrieving the latitude-longitude pair with the highest similarity, we are interested in retrieving a month-hour pair.
Thus, our time encoder (𝒯\\mathcal{T}) follows the exact same architecture as the location encoder (ℒ\\mathscr{L}). Similar to Eq. [1](https://arxiv.org/html/2507.10473v2#S3.E1 "Equation 1 ‣ 3.1 Image-Location Alignment ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), the time embedding is obtained from the proposed time representation using the following equation:

|     |     |     |     |
| --- | --- | --- | --- |
|  | Ti=𝒯⁡(Ui)=∑i=1Mfi​(γ⁡(unix2cyclic​(Ui),σi)).T\_{i}=\\mathcal{T}(U\_{i})=\\sum\_{i=1}^{M}f\_{i}(\\gamma(\\mathrm{\\texttt{unix2cyclic}}(U\_{i}),\\sigma\_{i})). |  | (5) |

### 3.3 Temporal Metric Learning

Visual features associated with location typically exhibit significant variation due to cultural, socio-economic and environmental factors, leading to abrupt changes in visual embeddings with respect to spatial distances. For instance, two neighborhoods within the same city often appear notably different. Therefore, standard contrastive objectives that clearly distinguish positives and negatives are suitable for aligning location and image embeddings (Eq. [2](https://arxiv.org/html/2507.10473v2#S3.E2 "Equation 2 ‣ 3.1 Image-Location Alignment ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")). In contrast, visual cues change more smoothly and continuously over time. As a result, labeling temporally adjacent samples as negatives, as is common in standard contrastive losses, can hinder effective learning. This makes it challenging to define explicit positives and negatives for time-image alignment. To address this, we propose a metric-learning objective called Temporal Metric Learning (TML), explicitly designed to leverage the smooth and cyclic nature of time by aligning instance similarity proportionally to the temporal difference.

Let’s consider the image embeddings {Vi}i=1B\\left\\{V\_{i}\\right\\}\_{i=1}^{B} and time embeddings {Ti}i=1B\\left\\{T\_{i}\\right\\}\_{i=1}^{B}. Instead of defining a set of positive and negative pairs for each embedding, we assign soft targets inversely proportional to the difference between the time associated to the image and time embeddings. Since months and hours are cyclical, using the regular ℓ2\\ell\_{2} distance between two normalized month-hour pairs (θi,ϕi)(\\theta\_{i},\\phi\_{i}) and (θj,ϕj)(\\theta\_{j},\\phi\_{j}) in an Euclidean space results in overestimated distance values, as shown in Figure [3](https://arxiv.org/html/2507.10473v2#S3.F3 "Figure 3 ‣ 3.1 Image-Location Alignment ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")(b). This problem can be solved by mapping the normalized month-hour pairs into the surface of a toroidal manifold, resulting in the new distance δi,j\\delta\_{i,j}:

|     |     |     |     |
| --- | --- | --- | --- |
|  | δi,j=∑α∈{θ,ϕ}min⁡(1−\|Δ​αi,j\|,\|Δ​αi,j\|)2.\\delta\_{i,j}=\\sqrt{\\sum\_{\\alpha\\in\\left\\{\\theta,\\phi\\right\\}}\\min(1-\|\\Delta\\alpha\_{i,j}\|,\|\\Delta\\alpha\_{i,j}\|)^{2}}. |  | (6) |

where Δ​αi,j=αi−αj\\Delta\\alpha\_{i,j}=\\alpha\_{i}-\\alpha\_{j}. Then, for each anchor image embedding ViV\_{i}, we compute a vector 𝐩i\\mathbf{p}\_{i} with the normalized cosine similarity scores with respect to all time embeddings TjT\_{j} in the batch. Similarly, we compute the vector 𝐪i\\mathbf{q}\_{i} with the normalized time difference between the anchor time and other times in the batch as follows:

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝐩i​\[j\]=exp⁡(Vi⋅Tj/τ)∑k=1Bexp⁡(Vi⋅Tk/τ),j∈\[B\],\\mathbf{p}\_{i}\[j\]=\\frac{\\exp\\left(V\_{i}\\cdot T\_{j}/\\tau\\right)}{\\sum\_{k=1}^{B}\\exp\\left(V\_{i}\\cdot T\_{k}/\\tau\\right)},\\quad j\\in\[B\], |  | (7) |

|     |     |     |     |
| --- | --- | --- | --- |
|  | 𝐪i​\[j\]=1−exp⁡(δi,j)∑k=1Bexp⁡(δi,k),j∈\[B\].\\mathbf{q}\_{i}\[j\]=1-\\frac{\\exp\\left(\\delta\_{i,j}\\right)}{\\sum\_{k=1}^{B}\\exp\\left(\\delta\_{i,k}\\right)},\\quad j\\in\[B\]. |  | (8) |

Since 𝐩i\\mathbf{p}\_{i} and 𝐪i\\mathbf{q}\_{i} are normalized, they have the same characteristics as probability mass functions. Thus, we define the image-time contrastive loss by computing the cross-entropy (CE\\mathrm{CE}) between 𝐩i\\mathbf{p}\_{i} and 𝐪i\\mathbf{q}\_{i}:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒt​i​m​e(i)=CE⁡(𝐩i,𝐪i).\\mathcal{L}\_{time}^{(i)}=\\mathrm{CE}(\\mathbf{p}\_{i},\\mathbf{q}\_{i}). |  | (9) |

The final training objective is defined as the sum of the image-location and image-time objectives, defined in Equations [2](https://arxiv.org/html/2507.10473v2#S3.E2 "Equation 2 ‣ 3.1 Image-Location Alignment ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") and [10](https://arxiv.org/html/2507.10473v2#S3.E10 "Equation 10 ‣ 3.3 Temporal Metric Learning ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"):

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ(i)=ℒl​o​c(i)+ℒt​i​m​e(i).\\mathcal{L}^{(i)}=\\mathcal{L}\_{loc}^{(i)}+\\mathcal{L}\_{time}^{(i)}. |  | (10) |

Table 1: Zero-shot time prediction on the unseen cameras of SkyFinder dataset. Rows marked by \* indicate methods we replicate, closely adhering to the protocols outlined by prior work.

| Method | Month | Hour | TPS↑\\uparrow |
|  | Error↓\\downarrow | Error↓\\downarrow |  |
| [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "")\* | 2.46 | 3.18 | 65.48 |
| [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "")\* | 1.62 | 3.61 | 71.42 |
| [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 "")\* | 2.72 | 3.05 | 63.25 |
| [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "")\\* w/ CLIP | 1.65 | 3.14 | 73.20 |
| [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "")\\* w/ CLIP | 1.62 | 2.98 | 74.06 |
| [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 "")\\* w/ CLIP | 1.56 | 2.87 | 75.02 |
| CLIP + cls. head | 1.61 | 3.17 | 73.37 |
| CLIP + reg. head | 1.55 | 3.06 | 74.33 |
| DINOv2 + cls. head | 2.24 | 3.74 | 65.61 |
| DINOv2 + reg. head | 2.12 | 3.53 | 67.49 |
| OpenCLIP + cls. head | 1.66 | 3.43 | 71.87 |
| OpenCLIP + reg. head | 1.72 | 3.34 | 71.75 |
| TimeLoc | 1.52 | 2.84 | 75.49 |
| GT-Loc (Ours) | 1.40 | 2.72 | 77.00 |

### 3.4 Inference

After training, the image, location, and time modalities share a unified embedding space. To predict the capture time and location of a query image IQI^{Q}, we compute the cosine similarity between its image embedding VQV^{Q} and the embeddings within the time gallery TGT^{G} and the location gallery LGL^{G}. The predicted time and location correspond to the gallery elements with the highest cosine similarities. For a visual representation of this inference process, refer to Supplementary Section [7](https://arxiv.org/html/2507.10473v2#S7 "7 Model inference ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space").

## 4 Experiments

We evaluate GT-Loc on both time-of-capture and geo-localization tasks, conduct ablation studies to justify key design choices, and assess robustness by measuring performance under limited training data and noisy annotations.

Datasets and evaluation details:
For training, we use two existing datasets: MediaEval Placing Tasks 2016 (MP-16) \[ [15](https://arxiv.org/html/2507.10473v2#bib.bib15 "")\] and Cross-View Time (CVT) \[ [26](https://arxiv.org/html/2507.10473v2#bib.bib26 "")\]. MP-16 consists of 4.72 Million images from Flickr annotated only with GPS coordinates. CVT originally consists of 206k geo-tagged smartphone pictures from the Yahoo Flickr Creative Commons 100 Million Dataset \[ [34](https://arxiv.org/html/2507.10473v2#bib.bib34 "")\] and 98k images from static outdoor webcams of the SkyFinder Dataset \[ [19](https://arxiv.org/html/2507.10473v2#bib.bib19 "")\]. Our zero-shot evaluation benchmark is constructed using a subset of images from SkyFinder that are not seen during training.

Geo-localization performance is evaluated by measuring the geodesic distance between the real and predicted GPS coordinates, and then computing the ratio of images that are correctly predicted within a threshold. Time prediction performance is evaluated by measuring the mean absolute ToY (ET​o​YE\_{ToY}) and ToD (ET​o​DE\_{ToD}) errors between the ground-truth and predicted times. We also report an overall Time Prediction Score (TPS) that combines both errors into a single metric. We compute the TPS based on our proposed cyclical time difference using the following equation:

|     |     |     |     |
| --- | --- | --- | --- |
|  | T​P​S=1−E~T​o​Y2+E~T​o​D22,TPS=1-\\sqrt{\\frac{\\widetilde{E}\_{ToY}^{2}+\\widetilde{E}\_{ToD}^{2}}{2}}, |  | (11) |

where E~T​o​Y,E~T​o​D∈\[0,1\]\\widetilde{E}\_{ToY},\\widetilde{E}\_{ToD}\\in\[0,1\] are the normalized time errors. A TPS of 1 represents a perfect prediction, while 0 represents the maximum possible cyclic error of 12 hours and 6 months. Please, refer to Supplementary Sections [8](https://arxiv.org/html/2507.10473v2#S8 "8 Implementation details ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") to [12](https://arxiv.org/html/2507.10473v2#S12 "12 Dataset details ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") for more details about the dataset, architecture and training protocol.

Table 2: Geo-localization accuracy on Im2GPS3k & GWS15k datasets, reported on the ratio of samples that are correctly predicted under a distance threshold of 1 km radius.

| Method | Im2GPS3k | GWS15k |
| --- | --- | --- |
| \[L\] kNN, sigma=4 ICCV’17\[ [38](https://arxiv.org/html/2507.10473v2#bib.bib38 "")\] | 7.2 | - |
| PlaNet ECCV’16\[ [41](https://arxiv.org/html/2507.10473v2#bib.bib41 "")\] | 8.5 | - |
| CPlaNet ECCV’18\[ [28](https://arxiv.org/html/2507.10473v2#bib.bib28 "")\] | 10.2 | - |
| ISNs ECCV’18\[ [20](https://arxiv.org/html/2507.10473v2#bib.bib20 "")\] | 10.5 | 0.1 |
| Translocator ECCV’22\[ [23](https://arxiv.org/html/2507.10473v2#bib.bib23 "")\] | 11.8 | 0.5 |
| GeoDecoder CVPR’23\[ [3](https://arxiv.org/html/2507.10473v2#bib.bib3 "")\] | 12.8 | 0.7 |
| GeoCLIP NeuRIPS’24\[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\] | 14.11 | 0.6 |
| PIGEOTTO CVPR’24\[ [5](https://arxiv.org/html/2507.10473v2#bib.bib5 "")\] | 11.3 | 0.7 |
| Img2Loc(LLaVA) SIGIR’24\[ [47](https://arxiv.org/html/2507.10473v2#bib.bib47 "")\] | 8.0 | - |
| GT-Loc (Ours) | 14.41 | 0.88 |

### 4.1 Comparison with Prior Methods

Time-of-capture prediction:
We present our time prediction results in Table [1](https://arxiv.org/html/2507.10473v2#S3.T1 "Table 1 ‣ 3.3 Temporal Metric Learning ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). Given the significant reproducibility challenges identified in prior work (see Supplementary Section [21](https://arxiv.org/html/2507.10473v2#S21 "21 Limitations and reproducibility challenges of existing time prediction methods ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")), we selected three representative baseline methods for a fair comparison with GT-Loc. The first baseline is the triple encoder architecture from [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 ""), which closely resembles our approach. Although they did not release code, pretrained weights, or exact dataset splits, their methodology is well-documented, allowing us to closely replicate their protocol. The second baseline is [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 ""), chosen due to its recent publication, public source code availability, and use of the CVT dataset—common to our evaluation. However, they do not provide the specific cross-camera split of CVT, and their original work only offers qualitative results. We extend their evaluation quantitatively for direct comparison. The final baseline is [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 ""), selected for its use of the SkyFinder dataset, with clear experimental procedures enabling straightforward replication despite the absence of source code. Importantly, all these baseline methods rely on geo-location metadata as input, unlike GT-Loc, which exclusively uses visual cues. Additionally, each baseline fully trains their models using different backbone architectures: [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "") employs InceptionV2, while both [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "") and [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 "") use DenseNet-121. To ensure fairness, we also evaluate scenarios in which these models use a frozen CLIP ViT-L/14 backbone, aligning closely with GT-Loc.

Our second set of baselines explores various frozen backbone architectures (e.g., CLIP ViT-L/14, DINOv2 ViT-L/14, and OpenCLIP ViT-G/14), paired with an MLP of matching capacity to our image encoder, and either regression or classification heads. These baselines illustrate the effectiveness of our retrieval-based approach compared to standard classification and regression techniques.

Lastly, we introduce TimeLoc, a robust baseline utilizing only the image and time encoders from GT-Loc, analogous to GeoCLIP \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\] and SatCLIP \[ [11](https://arxiv.org/html/2507.10473v2#bib.bib11 "")\] but specifically tailored for time prediction. Experimental results clearly demonstrate that GT-Loc achieves superior accuracy for both Time-of-Year (ToY) and Time-of-Day (ToD) predictions without additional metadata. Moreover, jointly training for time prediction and geo-localization enriches the learned temporal representations. In Figure [4](https://arxiv.org/html/2507.10473v2#S4.F4 "Figure 4 ‣ 4.1 Comparison with Prior Methods ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") , we show two qualitative examples from unseen cameras of the SkyFinder dataset, highlighting GT-Loc’s effectiveness in simultaneously predicting time and geo-location.

Table 3: Ablations for time prediction performance with different loss functions. ℓ2\\ell\_{2} refers to the Euclidean distance, while cyclic refers to the distance over the toroidal manifold.

| Loss | Month | Hour | TPS↓\\downarrow |
| Function | Error↓\\downarrow | Error↑\\uparrow |  |
| CLIP \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\] | 1.71 | 3.51 | 71.12 |
| RnC \[ [44](https://arxiv.org/html/2507.10473v2#bib.bib44 "")\] | 1.87 | 2.96 | 71.89 |
| SimCLR \[ [2](https://arxiv.org/html/2507.10473v2#bib.bib2 "")\] | 1.50 | 3.40 | 73.28 |
| TML (ℓ2\\ell\_{2}) | 1.53 | 2.74 | 75.88 |
| TML (Cyclic) | 1.40 | 2.72 | 77.00 |

![Refer to caption](https://arxiv.org/html/2507.10473v2/qualitative.png)Figure 4: (a) Sample images for two cameras of the SkyFinder test set with the ground truth location and capture time. (b) Spatial distribution of the predicted GPS coordinates colored by the cosine similarity between the location and image embeddings. (c) Histogram of the top-1k retrieved months and hours, weighted by the cosine similarity between the image and time embeddings (supp. Eq. [12](https://arxiv.org/html/2507.10473v2#S11.E12 "Equation 12 ‣ 11 Time-of-capture prediction histograms ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")). Both the top-1 predicted location and time are shown below the distributions.

Geo-localization: Table [2](https://arxiv.org/html/2507.10473v2#S4.T2 "Table 2 ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") presents the geo-localization performance of our model compared to recent expert methods. Overall, GT-Loc delivers competitive results compared against models like GeoCLIP on both datasets. It also outperforms the LLaVA-based Img2Loc variant, despite using at least 15 times fewer parameters.

### 4.2 Ablation Studies

In this section, we present additional experiments to evaluate the effectiveness of our proposed Temporal Metric Loss. Furthermore, in the supplementary material (Section [13](https://arxiv.org/html/2507.10473v2#S13 "13 Additional ablations ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")), we explore alternative image backbones (i.e., DINOv2 \[ [21](https://arxiv.org/html/2507.10473v2#bib.bib21 "")\], OpenCLIP \[ [7](https://arxiv.org/html/2507.10473v2#bib.bib7 "")\]), different time encoders (i.e., Time2Vec \[ [10](https://arxiv.org/html/2507.10473v2#bib.bib10 "")\], Circular Decomposition \[ [17](https://arxiv.org/html/2507.10473v2#bib.bib17 "")\]), and various temporal resolutions to provide additional insights and justification for our design choices.

We evaluate several alternative loss functions for time prediction. First, we implement a basic CLIP-based loss \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\]. Next, we explore a geo-localization-style contrastive loss that uses a dynamic queue and applies a false negative mask, excluding samples close to the anchor based on a specified threshold. We also experiment with the Rank-N-Contrast loss introduced by [Zha et al. \[44\]](https://arxiv.org/html/2507.10473v2#bib.bib44 ""), specifically designed for regression tasks by ranking samples according to their distances. Finally, we compare our proposed loss function against a variant using standard ℓ2\\ell\_{2} distance rather than cyclic temporal distance. The results in Table [3](https://arxiv.org/html/2507.10473v2#S4.T3 "Table 3 ‣ 4.1 Comparison with Prior Methods ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") demonstrate that our Temporal Metric Loss significantly outperforms all other evaluated losses for both Time-of-Day (ToD) and Time-of-Year (ToY) predictions.

Table 4: Impact of Limited Data on the Robustness of Time Prediction

| Data | Month | Hour | TPS↑\\uparrow |
| --- | --- | --- | --- |
| Availability | Error↓\\downarrow | Error↓\\downarrow |  |
| 100% | 1.40 | 2.72 | 77.00 |
| 50% | 1.69 | 2.83 | 74.02 |
| 10% | 1.70 | 2.94 | 73.51 |
| 5% | 1.89 | 2.86 | 72.07 |

Table 5: Ablations for robustness to label noise for time-prediction.

| Label | Month | Hour | TPS↑\\uparrow |
| --- | --- | --- | --- |
| Noise (σ\\sigma) | Error↓\\downarrow | Error↓\\downarrow |  |
| 0 | 1.40 | 2.72 | 77.00 |
| 1 | 1.52 | 2.71 | 76.00 |
| 2 | 1.75 | 2.74 | 73.81 |
| 3 | 2.16 | 2.72 | 69.92 |

### 4.3 Robustness of GT-Loc

Timestamp‐prediction datasets are often scarce or plagued by missing and noisy metadata, so robustness is essential. We first measure performance as we shrink the training set from 100% to 5% in four stages, then simulate label noise by adding Gaussian perturbations with standard deviation between σ∈\[0,3\]\\sigma\\in\[0,3\] months and hours to the training annotations. As Tables [4](https://arxiv.org/html/2507.10473v2#S4.T4 "Table 4 ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") and [5](https://arxiv.org/html/2507.10473v2#S4.T5 "Table 5 ‣ 4.2 Ablation Studies ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") show, GT-Loc’s errors rise only modestly by 0.3 months and 0.22 hours, even with just 5% of the data, and it remains stable up to σ=2\\sigma=2, with significant degradation appearing only at σ≥3\\sigma\\geq 3. These results confirm GT-Loc’s resilience under realistic data constraints.

### 4.4 Compositional Image Retrieval

In this section, we evaluate GT-Loc’s capability for compositional image retrieval tasks, leveraging its unified multimodal embedding space. Specifically, we explore retrieving images based on joint queries consisting of both location and time information. Given query location LQL^{Q} and time TQT^{Q}, we generate a multimodal representation by averaging their embeddings, inspired by multimodal retrieval methods proposed in prior works \[ [31](https://arxiv.org/html/2507.10473v2#bib.bib31 ""), [32](https://arxiv.org/html/2507.10473v2#bib.bib32 "")\].

To provide meaningful baselines for comparison, we selected the method by [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 ""), which is conceptually similar but was originally intended only for time and location classification tasks. Since their original work did not consider compositional retrieval explicitly, we adapt their model to our retrieval scenario as described in Supplementary section [14](https://arxiv.org/html/2507.10473v2#S14 "14 Compositional image retrieval details ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). We evaluate retrieval performance using recall metrics at ranks 1, 5, and 10. A retrieved image is considered correct if its ground truth timestamp is within one hour and one month, respectively, of the query, and if its location is within 25 km.

Table [1](https://arxiv.org/html/2507.10473v2#S3.T1 "Table 1 ‣ 3.3 Temporal Metric Learning ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") shows quantitative results for these baselines. GT-Loc consistently achieves higher recall across all ranks, significantly outperforming other methods. This indicates that our unified retrieval-based embedding approach effectively encodes joint geo-temporal information, providing richer representations for retrieval tasks. Qualitative examples demonstrating GT-Loc’s compositional retrieval capabilities are shown in Supplementary section [14](https://arxiv.org/html/2507.10473v2#S14 "14 Compositional image retrieval details ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space").

Table 6: Zero-shot composed retrieval (T+L→IT+L\\rightarrow I) on the unseen cameras of the SkyFinder dataset.

| Method | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "")\* | 0.91 | 7.81 | 13.61 |
| [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "")\\* w/ CLIP | 2.58 | 16.22 | 29.46 |
| GT-Loc | 6.69 | 24.58 | 38.54 |

### 4.5 Qualitative results using text queries

We also investigate the ability of GT-Loc to use the pretrained CLIP text encoder to retrieve times and locations mentioned in the text. For this task, we follow GeoCLIP’s approach and replace the image backbone by a text backbone, keeping the trained MLP, location encoder and time encoder. For each text, we create a text embedding, pass it through the MLP and compare it against the location and time galleries. We then create spatial and temporal distributions of the top retrieved samples for each modality, as shown Figure [5](https://arxiv.org/html/2507.10473v2#S4.F5 "Figure 5 ‣ 4.5 Qualitative results using text queries ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), where we see that not only is our model able to accurately pinpoint the location, but it also creates meaningful time distributions to words such as “winter” and “evening” that do not explicitly mention the time.

![Refer to caption](https://arxiv.org/html/2507.10473v2/text.png)Figure 5: Qualitative examples of geo-localization and time-of-capture prediction using text queries. Top: prompt passed to CLIP’s text encoder. Middle: spatial distribution of the predicted geo-locations with the highest cosine similarity. Bottom: histogram of the top-1k predicted months and hours with the highest cosine similarity. GT-Loc is capable of providing good estimates of the time and GPS coordinates to each of the text queries, even when time is not explicitly specified.

## 5 Conclusion

We introduce GT-Loc, a novel framework for jointly predicting the time and location of an image using a retrieval approach. GT-Loc not only shows competitive performance compared to state-of-the-art geo-localization models but also introduces the capability of precise time-of-capture predictions. A key innovation of our approach is the novel temporal metric loss, which significantly outperforms traditional contrastive losses in time prediction tasks.

Furthermore, our results demonstrate that GT-Loc extends beyond standard time-of-capture prediction and geo-localization tasks. It supports additional functionalities like compositional image retrieval (T+L→IT+L\\rightarrow I), as well as text-to-location and text-to-image retrieval, indicating a profound understanding of the interplay between images, locations, and time.

## 6 Acknowledgements

Supported by Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0074. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

## References

- \[1\]
Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude, Elliot Vincent, Lintao Xu, Hongyu Zhou, and Loic Landrieu.
OpenStreetView-5M: The many roads to global visual geolocation.
_CVPR_, 2024.

- \[2\]
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
A simple framework for contrastive learning of visual representations.
In _International conference on machine learning_, pages 1597–1607. PMLR, 2020.

- \[3\]
Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda, and Mubarak Shah.
Where we are and what we’re looking at: Query based worldwide image geo-localization using hierarchies and scenes.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 23182–23190, 2023.

- \[4\]
Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra.
Imagebind: One embedding space to bind them all.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 15180–15190, 2023.

- \[5\]
Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn.
Pigeon: Predicting image geolocations.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, pages 12893–12902, 2024.

- \[6\]
Yannick Hold-Geoffroy, Kalyan Sunkavalli, Sunil Hadap, Emiliano Gambaretto, and Jean-François Lalonde.
Deep outdoor illumination estimation.
In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 7312–7321, 2017.

- \[7\]
Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt.
Openclip, 2021.
If you use this software, please cite it as below.

- \[8\]
Nathan Jacobs, Scott Satkin, Nathaniel Roman, Richard Speyer, and Robert Pless.
Geolocating static cameras.
In _2007 IEEE 11th International Conference on Computer Vision_, pages 1–6. IEEE, 2007.

- \[9\]
Nathan Jacobs, Walker Burgin, Nick Fridrich, Austin Abrams, Kylia Miskell, Bobby H. Braswell, Andrew D. Richardson, and Robert Pless.
The global network of outdoor webcams: Properties and applications.
In _ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL)_, pages 111–120, 2009.

- \[10\]
Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, and Marcus Brubaker.
Time2vec: Learning a vector representation of time.
_arXiv preprint arXiv:1907.05321_, 2019.

- \[11\]
Konstantin Klemmer, Esther Rolf, Caleb Robinson, Lester Mackey, and Marc Rußwurm.
Satclip: Global, general-purpose location embeddings with satellite imagery.
_arXiv preprint arXiv:2311.17179_, 2023.

- \[12\]
Parth Parag Kulkarni, Gaurav Kumar Nayak, and Mubarak Shah.
Cityguessr: City-level video geo-localization on a global scale.
In _European Conference on Computer Vision_, pages 293–311. Springer, 2024.

- \[13\]
Jean-François Lalonde, Srinivasa G Narasimhan, and Alexei A Efros.
What do the sun and the sky tell us about the camera?
_International Journal of Computer Vision_, 88:24–51, 2010.

- \[14\]
Jean-François Lalonde, Alexei A Efros, and Srinivasa G Narasimhan.
Estimating the natural illumination conditions from a single outdoor image.
_International Journal of Computer Vision_, 98:123–145, 2012.

- \[15\]
Martha Larson, Mohammad Soleymani, Guillaume Gravier, Bogdan Ionescu, and Gareth JF Jones.
The benchmarking initiative for multimedia evaluation: Mediaeval 2016.
_IEEE MultiMedia_, 24(1):93–96, 2017.

- \[16\]
Xiaopeng Li, Wenyuan Xu, Song Wang, and Xianshan Qu.
Are you lying: Validating the time-location of outdoor images.
In _Applied Cryptography and Network Security: 15th International Conference, ACNS 2017, Kanazawa, Japan, July 10-12, 2017, Proceedings 15_, pages 103–123. Springer, 2017.

- \[17\]
Oisin Mac Aodha, Elijah Cole, and Pietro Perona.
Presence-only geographical priors for fine-grained image classification.
In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, 2019.

- \[18\]
Gengchen Mai, Ni Lao, Yutong He, Jiaming Song, and Stefano Ermon.
CSP: Self-supervised contrastive spatial pre-training for geospatial-visual representations.
In _Proceedings of the 40th International Conference on Machine Learning_, pages 23498–23515. PMLR, 2023.

- \[19\]
Radu Paul Mihail, Scott Workman, Zach Bessinger, and Nathan Jacobs.
Sky segmentation in the wild: An empirical study.
In _IEEE Winter Conference on Applications of Computer Vision (WACV)_, pages 1–6, 2016.

- \[20\]
Eric Muller-Budack, Kader Pustu-Iren, and Ralph Ewerth.
Geolocation estimation of photos using a hierarchical model and scene classification.
In _Proceedings of the European conference on computer vision (ECCV)_, pages 563–579, 2018.

- \[21\]
Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski.
Dinov2: Learning robust visual features without supervision, 2023.

- \[22\]
Rafael Padilha, Tawfiq Salem, Scott Workman, Fernanda A Andaló, Anderson Rocha, and Nathan Jacobs.
Content-aware detection of temporal metadata manipulation.
_IEEE Transactions on Information Forensics and Security_, 17:1316–1327, 2022.

- \[23\]
Shraman Pramanick, Ewa M Nowara, Joshua Gleason, Carlos D Castillo, and Rama Chellappa.
Where in the world is this image? transformer-based geo-localization in the wild.
In _European Conference on Computer Vision_, pages 196–215. Springer, 2022.

- \[24\]
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
Learning transferable visual models from natural language supervision.
In _International conference on machine learning_, pages 8748–8763. PMLR, 2021.

- \[25\]
Krishna Regmi and Mubarak Shah.
Bridging the domain gap for ground-to-aerial image matching.
In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 470–479, 2019.

- \[26\]
Tawfiq Salem, Scott Workman, and Nathan Jacobs.
Learning a dynamic map of visual appearance.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 12435–12444, 2020.

- \[27\]
Tawfiq Salem, Jisoo Hwang, and Rafael Padilha.
Timestamp estimation from outdoor scenes.
In _Annual ADFSL Conference on Digital Forensics, Security and Law_, number 2. Embry-Riddle Aeronautical University Commons, 2022.

- \[28\]
Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han.
Cplanet: Enhancing image geolocalization by combinatorial partitioning of maps.
In _Proceedings of the European Conference on Computer Vision (ECCV)_, pages 536–551, 2018.

- \[29\]
Yujiao Shi, Liu Liu, Xin Yu, and Hongdong Li.
Spatial-aware feature aggregation for image based cross-view geo-localization.
_Advances in Neural Information Processing Systems_, 32, 2019.

- \[30\]
Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li.
Where am i looking at? joint location and orientation estimation by cross-view matching.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 4064–4072, 2020.

- \[31\]
Nina Shvetsova, Brian Chen, Andrew Rouditchenko, Samuel Thomas, Brian Kingsbury, Rogerio S Feris, David Harwath, James Glass, and Hilde Kuehne.
Everything at once-multi-modal fusion transformer for video retrieval.
In _Proceedings of the ieee/cvf conference on computer vision and pattern recognition_, pages 20020–20029, 2022.

- \[32\]
Sirnam Swetha, Mamshad Nayeem Rizve, Nina Shvetsova, Hilde Kuehne, and Mubarak Shah.
Preserving modality structure improves multi-modal learning.
In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 21993–22003, 2023.

- \[33\]
Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng.
Fourier features let networks learn high frequency functions in low dimensional domains.
_Advances in neural information processing systems_, 33:7537–7547, 2020.

- \[34\]
Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li.
Yfcc100m: The new data in multimedia research.
_Communications of the ACM_, 59(2):64–73, 2016.

- \[35\]
Aysim Toker, Qunjie Zhou, Maxim Maximov, and Laura Leal-Taixé.
Coming down to earth: Satellite-to-street view synthesis for geo-localization.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 6488–6497, 2021.

- \[36\]
Tsung-Hung Tsai, Wei-Cih Jhou, Wen-Huang Cheng, Min-Chun Hu, I-Chao Shen, Tekoing Lim, Kai-Lung Hua, Ahmed Ghoneim, M Anwar Hossain, and Shintami C Hidayati.
Photo sundial: estimating the time of capture in consumer photos.
_Neurocomputing_, 177:529–542, 2016.

- \[37\]
Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah.
Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization.
_Advances in Neural Information Processing Systems_, 36, 2024.

- \[38\]
Nam Vo, Nathan Jacobs, and James Hays.
Revisiting im2gps in the deep learning era.
In _Proceedings of the IEEE international conference on computer vision_, pages 2621–2630, 2017.

- \[39\]
L Warszawski, K Frieler, V Huber, F Piontek, O Serdeczny, X Zhang, Q Tang, M Pan, Y Tang, Q Tang, et al.
Center for international earth science information network—ciesin—columbia university.(2016). gridded population of the world, version 4 (gpwv4): Population density. palisades. ny: Nasa socioeconomic data and applications center (sedac). doi: 10. 7927/h4np22dq.
_Atlas of environmental risks facing China under climate change_, page 228, 2017.

- \[40\]
Scott Wehrwein, Kavita Bala, and Noah Snavely.
Shadow detection and sun direction in photo collections.
In _2015 International Conference on 3D Vision_, pages 460–468. IEEE, 2015.

- \[41\]
Tobias Weyand, Ilya Kostrikov, and James Philbin.
Planet-photo geolocation with convolutional neural networks.
In _Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14_, pages 37–55. Springer, 2016.

- \[42\]
Scott Workman, Richard Souvenir, and Nathan Jacobs.
Wide-area image geolocalization with aerial reference imagery.
In _IEEE International Conference on Computer Vision (ICCV)_, pages 1–9, 2015.
Acceptance rate: 30.3%.

- \[43\]
Angelos Zavras, Dimitrios Michail, Begüm Demir, and Ioannis Papoutsis.
Mind the modality gap: Towards a remote sensing vision-language model via cross-modal alignment.
_arXiv preprint arXiv:2402.09816_, 2024.

- \[44\]
Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, and Dina Katabi.
Rank-n-contrast: Learning continuous representations for regression.
_Advances in Neural Information Processing Systems_, 36, 2024.

- \[45\]
Menghua Zhai, Tawfiq Salem, Connor Greenwell, Scott Workman, Robert Pless, and Nathan Jacobs.
Learning geo-temporal image features.
In _British Machine Vision Conference_, 2019.

- \[46\]
Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba.
Places: A 10 million image database for scene recognition.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_, 2017.

- \[47\]
Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu, Sheng Li, and Gengchen Mai.
Img2loc: Revisiting image geolocalization using multi-modality foundation models and image-based retrieval-augmented generation.
In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_, pages 2749–2754, 2024.

- \[48\]
Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al.
Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment.
_arXiv preprint arXiv:2310.01852_, 2023.

- \[49\]
Sijie Zhu, Taojiannan Yang, and Chen Chen.
Vigor: Cross-view image geo-localization beyond one-to-one retrieval.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 3640–3649, 2021.

- \[50\]
Sijie Zhu, Mubarak Shah, and Chen Chen.
Transgeo: Transformer is all you need for cross-view image geo-localization.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 1162–1171, 2022.


Supplementary Material

## 7 Model inference

![Refer to caption](https://arxiv.org/html/2507.10473v2/teaser.png)Figure 6: GT-Loc during inference.

Our framework consists of a model that can predict both the location and capture-time of an image at the same time using a retrieval approach. Given a query image IQI^{Q}, a gallery of GPS coordinates and a gallery of timestamps, GT-Loc maps the three modalities into a shared feature space using an image, location and time encoder. The query image embedding VQV^{Q} is compared against a set of location embeddings LGL^{G} and time embeddings TGT^{G}. The GPS and timestamp with the highest cosine similarity are selected as the predictions of our model. Figure [6](https://arxiv.org/html/2507.10473v2#S7.F6 "Figure 6 ‣ 7 Model inference ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") represents the overview of our approach.

## 8 Implementation details

Following GeoCLIP, the backbone of the image encoder is a pretrained ViT-L/14 from CLIP and the MLP consists of two fully connected layers with the ReLU activation function and dimensions 768 and 512 respectively. We use the same architecture for the time and location encoders as GeoCLIP. Both employ three RFF positional encoding layers, mapping the 2-dimensional GPS to a vector with 512 dimensions. The standard deviation values used to sample the RFF are σi∈{20,24,28}\\sigma\_{i}\\in\\{2^{0},2^{4},2^{8}\\}. The MLPs from the time and location encoder have three hidden layers with 1024 dimensions and a projection layer to map the final embeddings into a feature space of 512 dimensions. In the location encoder, we use a dynamic queue that stores the last 4096 seen locations, but we don’t use it for time. The GPS coordinates and times are augmented by adding Gaussian noise with standard deviation of 150 meters for the in-batch GPS, 1500 meters for the GPS queue, 0.15 months and 0.15 hours for time. We perform two augmentations for each image in the training set using random resized crops of size 224, random horizontal flipping and image normalization.

## 9 Training protocol

GT-Loc is trained for 20 epochs using a cosine decay scheduler, with learning rate values ranging from αm​a​x=3×10−5\\alpha\_{max}=3\\times 10^{-5} to αm​i​n=3×10−7\\alpha\_{min}=3\\times 10^{-7}. We use Adam optimizer with coefficients β1=0.9\\beta\_{1}=0.9, β2=0.999\\beta\_{2}=0.999 and ℓ2\\ell\_{2} penalty of 1×10−61\\times 10^{-6}. For the contrastive losses, we use two learnable temperature parameters that are optimized during training. The batch size BB is set to 512 for all experiments, and the models are trained on a machine with 12 CPU cores and a NVIDIA RTX A6000 GPU.

## 10 Additional qualitative results

We show additional qualitative results of our method in figures [7](https://arxiv.org/html/2507.10473v2#S10.F7 "Figure 7 ‣ 10 Additional qualitative results ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") and [8](https://arxiv.org/html/2507.10473v2#S10.F8 "Figure 8 ‣ 10 Additional qualitative results ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). We include a failure case, on the last row of figure [8](https://arxiv.org/html/2507.10473v2#S10.F8 "Figure 8 ‣ 10 Additional qualitative results ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), where the time error is high because of the presence of fog in the image.

![Refer to caption](https://arxiv.org/html/2507.10473v2/supp_text_figure.png)Figure 7: Additional qualitative examples of geo-localization and time-of-capture prediction using text queries. Top: prompt passed to CLIP’s text encoder. Middle: spatial distribution of the predicted geo-locations with the highest cosine similarity. Bottom: histogram of the predicted months and hours with the highest cosine similarity.![Refer to caption](https://arxiv.org/html/2507.10473v2/qual_results_updated.png)Figure 8: (a) Additional sample predictions for three cameras of the SkyFinder test set with the ground truth location and capture time. (b) Spatial distribution of the predicted GPS coordinates colored by the cosine similarity between the location and image embeddings. (c) Temporal distribution of the predicted month and hour, weighted by the cosine similarity between the time and image embeddings. The red bin contains the top-1 predicted time.

## 11 Time-of-capture prediction histograms

The time prediction histograms, shown in figures [4](https://arxiv.org/html/2507.10473v2#S4.F4 "Figure 4 ‣ 4.1 Comparison with Prior Methods ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), [5](https://arxiv.org/html/2507.10473v2#S4.F5 "Figure 5 ‣ 4.5 Qualitative results using text queries ‣ 4 Experiments ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), [7](https://arxiv.org/html/2507.10473v2#S10.F7 "Figure 7 ‣ 10 Additional qualitative results ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), and [8](https://arxiv.org/html/2507.10473v2#S10.F8 "Figure 8 ‣ 10 Additional qualitative results ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), are computed using the following equation:

|     |     |     |     |
| --- | --- | --- | --- |
|  | Ci=∑j=1NG𝟙\[j∈ℬi\]⋅IQ⋅TGj,C\_{i}=\\sum\_{j=1}^{N\_{G}}\\mathbbm{1}\_{\[j\\in\\mathcal{B}\_{i}\]}\\cdot I^{Q}\\cdot T^{G}\_{j}, |  | (12) |

where ℬi\\mathcal{B}\_{i} is the set of gallery embeddings that correspond to the iith bin, CiC\_{i} is the bin count, NGN\_{G} is the gallery size, IQI^{Q} is the query image embedding, TjGT^{G}\_{j} is the jjth time embedding from the gallery, and 𝟙\\mathbbm{1} is an indicator variable.

## 12 Dataset details

We apply two filters to remove samples from the CVT that don’t provide meaningful temporal information. In particular, we remove all night-time and indoor images, since they often have inconsistent temporal cues. To remove night images, we estimate the sunrise and sunset times from the date, latitude and longitude using the General Solar Position algorithm, and remove all samples before sunrise or after sunset. Then, for indoor images, we leverage a CNN model pretrained on the Places365 Dataset \[ [46](https://arxiv.org/html/2507.10473v2#bib.bib46 "")\]. In general, night images often have inconsistent artificial lighting, more noise or specialized cameras such as night vision. Indoor images also have artificial lighting and controlled temperature, making it difficult to estimate the time or date.

Regarding the levels of noise in the dataset, the SkyFinder subset consists of images with accurate time estimates, since they were collected from calibrated outdoor webcams. However, we observed that CVT has a moderate amount of noisy labels. Thus in order to train a model that can accurately predict the time, we need both datasets in the training set.

For evaluating the models, we employ a subset of unseen SkyFinder cameras, as well as two geo-localization datasets used by other state-of-the-art methods for evaluation: Im2GPS3k and GWS15k. Similar to GeoCLIP, we create a 100k GPS gallery to evaluate the model on Im2GPS3k, a 500k GPS gallery for GWS15k, and a 100k time gallery for the SkyFinder test set. The GPS galleries are created by sampling GPS coordinates from the MP-16 dataset, while the time gallery is created by sampling times from the combined CVT and SkyFinder training sets.

## 13 Additional ablations

### 13.1 Image backbones

To evaluate the impact of different image embeddings on time prediction performance, we conducted ablation studies using three backbones: DINOv2-L \[ [21](https://arxiv.org/html/2507.10473v2#bib.bib21 "")\], OpenCLIP ViT-G \[ [7](https://arxiv.org/html/2507.10473v2#bib.bib7 "")\], and OpenAI’s original CLIP ViT-L \[ [24](https://arxiv.org/html/2507.10473v2#bib.bib24 "")\]. For these experiments, we used the TimeLoc variant of our model, which incorporates only the image and time encoders. The results, summarized in Table [7](https://arxiv.org/html/2507.10473v2#S13.T7 "Table 7 ‣ 13.1 Image backbones ‣ 13 Additional ablations ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), indicate that OpenAI’s CLIP ViT-L achieves the lowest errors for both hours and months, as well as the highest Time Prediction Score (TPS).

| Backbone | Param. | Month | Hour | TPS |
| --- | --- | --- | --- | --- |
|  |  | Error | Error |  |
| DINOv2-L | 0.3B | 2.10 | 3.25 | 68.71 |
| OpenCLIP-G | 1.8B | 1.57 | 2.94 | 74.65 |
| CLIP-L | 0.2B | 1.52 | 2.84 | 75.49 |

Table 7: Comparison of time prediction performance using different frozen image backbones.

### 13.2 Time representation

Motivated by the cyclical nature of time, [Mac Aodha et al. 17](https://arxiv.org/html/2507.10473v2#bib.bib17 "") used circular decomposition to wrap the temporal input to their geographical prior encoder, resulting in similar embeddings for dates that are close to the start and end of the year, such as December 31s​t31^{st} and January 1s​t1^{st}. To achieve this,
for each dimension ll of the temporal input x→\\vec{x}, they perform the mapping \[sin⁡(π​xl),cos⁡(π​xl)\]\[\\sin(\\pi x^{l}),\\cos(\\pi x^{l})\], resulting in two numbers for each dimension.

Time2Vec\[ [10](https://arxiv.org/html/2507.10473v2#bib.bib10 "")\] is a method for encoding time that captures both periodic and non-periodic patterns. It transforms scalar time values into a vector of size k+1k+1. The first element models linear, non-periodic trends, while the remaining elements are defined by a periodic activation function (e.g., sine), capturing repeating temporal behaviors like daily or weekly cycles. The representation is defined as:

|     |     |     |
| --- | --- | --- |
|  | t​2​v​(τ)​\[i\]={ωi​τ+ϕiif ​i=0,F⁡(ωi​τ+ϕi)if ​1≤i≤k,t2v(\\tau)\[i\]=\\begin{cases}\\omega\_{i}\\tau+\\phi\_{i}&\\text{if }i=0,\\\<br>F(\\omega\_{i}\\tau+\\phi\_{i})&\\text{if }1\\leq i\\leq k,\\end{cases} |  |

where FF is a periodic activation function, typically sin\\sin, and ωi\\omega\_{i} and ϕi\\phi\_{i} are learnable parameters representing the frequency and phase shift, respectively.

Table 8: Ablations for time prediction performance using different time encoders.

| Time | Month | Hour | TPS |
| --- | --- | --- | --- |
| encoder | Error | Error |  |
| Circular decomp. | 1.59 | 2.86 | 74.80 |
| Time2Vec | 1.56 | 2.62 | 75.99 |
| RFF | 1.40 | 2.72 | 77.00 |

### 13.3 Time-of-year scale

Time-of-year (ToY) can be represented at either a monthly or daily scale. In practice, the choice of time scale should not significantly affect the results, as the value is normalized before being passed to the time encoder, 𝒯⁡(⋅)\\mathcal{T}(\\cdot). However, two approaches are available. The first approach converts the integer month mim\_{i} and day did\_{i} into a real-valued month, normalized over a 12-month period, as shown in Equations [3](https://arxiv.org/html/2507.10473v2#S3.E3 "Equation 3 ‣ 3.2 Time Representation ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") and [4](https://arxiv.org/html/2507.10473v2#S3.E4 "Equation 4 ‣ 3.2 Time Representation ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"). The second approach represents ToY as the number of days elapsed since the start of the year, normalized over 365 days (assuming no leap years in our dataset). This representation is defined as:

|     |     |     |
| --- | --- | --- |
|  | θi=1365​(di−1+∑k=1i𝒟⁡(mk−1)),\\theta\_{i}=\\frac{1}{365}\\left(d\_{i}-1+\\sum\_{k=1}^{i}\\mathcal{D}(m\_{k}-1)\\right), |  |

where 𝒟⁡(mk)\\mathcal{D}(m\_{k}) is the number of days in month mkm\_{k}, and it is assumed that 𝒟⁡(0)=0\\mathcal{D}(0)=0. In Table [9](https://arxiv.org/html/2507.10473v2#S13.T9 "Table 9 ‣ 13.3 Time-of-year scale ‣ 13 Additional ablations ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), we empirically show that using a monthly scale for ToY representation results in slightly better performance. However, we attribute this improvement to statistical noise rather than the time representation method itself.

Table 9: Ablations for time prediction performance using monthly and daily scales.

| ToY | Month | Hour | TPS |
| --- | --- | --- | --- |
| Scale | Error | Error |  |
| daily | 1.45 | 2.71 | 76.61 |
| monthly | 1.40 | 2.72 | 77.00 |

## 14 Compositional image retrieval details

To compare GT-Loc against a suitable baseline, we repurpose the model proposed by [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "") for compositional retrieval. In its original form, the model comprises an image encoder CI​(I)C\_{I}(I), a time encoder CT​(t)C\_{T}(t), and a location encoder CL​(L)C\_{L}(L). It concatenates the image and time embeddings to predict location via P⁡(l∣CI,CT)P(l\\mid C\_{I},C\_{T}), and similarly concatenates the image and location embeddings to predict time via P⁡(t∣CI,CL)P(t\\mid C\_{I},C\_{L}).

To adapt it for compositional image retrieval, we take a query time TQT^{Q} and location LQL^{Q} along with an image gallery IG{I^{G}}. First, we concatenate each image embedding CI​(IG)C\_{I}(I^{G}) with CT​(TQ)C\_{T}(T^{Q}) or CL​(LQ)C\_{L}(L^{Q}) and feed these pairs to the respective classification heads. This yields probability distributions Pi​(l∣CI,CT)P\_{i}(l\\mid C\_{I},C\_{T}) and Pi​(t∣CI,CL)P\_{i}(t\\mid C\_{I},C\_{L}) for each image ii in the gallery, indicating how well that image matches the queried location and time. Since we already know the desired location and time, we extract the corresponding probabilities from the distributions and average them to produce a final similarity score. Ranking by this score allows us to retrieve the top-kk gallery images most likely to match (TQ,LQ)(T^{Q},L^{Q}). This adaptation of the [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "") model provides a direct, fair comparison to GT-Loc’s performance on compositional retrieval tasks.

Figure [9](https://arxiv.org/html/2507.10473v2#S14.F9 "Figure 9 ‣ 14 Compositional image retrieval details ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") shows two qualitative results of our compositional retrieval model.

![Refer to caption](https://arxiv.org/html/2507.10473v2/comp_ret.png)Figure 9: Illustrating compositionalL+T→IL+T\\rightarrow{I} retrieval with GT-Loc. Each example Left: query location and query time, showing actual image. Right: retrieved image for given query location and time.

## 15 Additional geo-localization analysis

Figure [10](https://arxiv.org/html/2507.10473v2#S15.F10 "Figure 10 ‣ 15 Additional geo-localization analysis ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") presents the cumulative geolocation error evaluated over a range of distance thresholds. In addition to our main results, we include evaluations on two extra datasets: YFCC26k and OSV-5M. These plots allow for a more comprehensive comparison of model performance across diverse data distributions. We also include a comparison against the hybrid method proposed by [Astruc et al. \[1\]](https://arxiv.org/html/2507.10473v2#bib.bib1 ""), which was not shown in the main paper. We observe that our method, GT-Loc, consistently outperforms [Astruc et al. \[1\]](https://arxiv.org/html/2507.10473v2#bib.bib1 "") on all datasets with the exception of OSV-5M. We attribute this to the fact that OSV-5M contains imagery that is in-domain for their model, whereas it is out-of-domain for ours.

![Refer to caption](https://arxiv.org/html/2507.10473v2/cumulative_errors_log.png)Figure 10: Cumulative geolocation error at different thresholds.

## 16 Effect of population density on geo-localization

Geo-localization datasets, particularly those collected via web-scraping or social media platforms, are inherently biased toward regions with higher population density. This raises a natural question: how does the performance of geo-localization algorithms vary between densely and sparsely populated areas? To investigate this, we conduct an additional analysis. For each image in our dataset, we obtain the corresponding population density from the Gridded Population of the World version 4 (GPWv4) dataset \[ [39](https://arxiv.org/html/2507.10473v2#bib.bib39 "")\]. We then group the data into ranges based on population density (measured in p​o​p/m2pop/m^{2}) and compute the median geo-localization error within each group. This stratified evaluation allows us to assess model performance across regions with varying levels of human activity. Consistent with the findings of [Astruc et al. \[1\]](https://arxiv.org/html/2507.10473v2#bib.bib1 ""), we observe that geo-localization errors tend to be higher in areas with lower population density, as summarized in Table [10](https://arxiv.org/html/2507.10473v2#S16.T10 "Table 10 ‣ 16 Effect of population density on geo-localization ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space").

Table 10: Median geo-localization error by population density group.

| Pop. density | Im2GPS3k (km) | GWS15k (km) |
| --- | --- | --- |
| <100<100 | 266.66 | 1244.22 |
| \[100,1​k)\[100,1\\text{k}) | 246.12 | 818.65 |\
| \[1​k,10​k)\[1\\text{k},10\\text{k}) | 197.99 | 639.73 |\
| ≥10​k\\geq 10\\text{k} | 38.32 | 1035.81 |\
\
## 17 Model performance vs. gallery size\
\
Figure [11](https://arxiv.org/html/2507.10473v2#S17.F11 "Figure 11 ‣ 17 Model performance vs. gallery size ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") shows how the size of the gallery affects geo-localization and time prediction performance. We observe that increasing the size of the gallery leads to improved results, but the gains tend to saturate quickly. In particular, a gallery containing 100,000 samples is already sufficient to capture the majority of the performance improvements for geo-localization. Interestingly, for time-of-capture prediction, we find that even smaller archives, such as those with only 4,000 samples, can still yield strong performance.\
\
![Refer to caption](https://arxiv.org/html/2507.10473v2/perf_vs_gal.png)Figure 11: Effect of gallery size on geo-localization and time prediction.\
\
## 18 Analysis of the learned embedding space\
\
One of the key motivations for GT-Loc is to align images, time, and location in a shared multimodal embedding space. This approach is inspired by prior works like GeoCLIP \[ [37](https://arxiv.org/html/2507.10473v2#bib.bib37 "")\], SatCLIP \[ [11](https://arxiv.org/html/2507.10473v2#bib.bib11 "")\], and CSP \[ [18](https://arxiv.org/html/2507.10473v2#bib.bib18 "")\], which embed images and GPS coordinates in shared spaces, as well as methods like ImageBind \[ [4](https://arxiv.org/html/2507.10473v2#bib.bib4 "")\], LanguageBind \[ [48](https://arxiv.org/html/2507.10473v2#bib.bib48 "")\], Everything At Once \[ [31](https://arxiv.org/html/2507.10473v2#bib.bib31 "")\], and Preserving Modality \[ [32](https://arxiv.org/html/2507.10473v2#bib.bib32 "")\], which align multiple modalities such as images, text, videos, and audio. Our work extends this idea to include temporal information, showing in Table [1](https://arxiv.org/html/2507.10473v2#S3.T1 "Table 1 ‣ 3.3 Temporal Metric Learning ‣ 3 Method ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") that aligning these three modalities leads to improved time prediction performance compared to using only images and time.\
\
To explore the relationships between these modalities in the learned embedding space, we performed Principal Component Analysis (PCA) on the embeddings. While PCA has limitations in fully capturing the underlying structure of high-dimensional spaces, the results provide interesting qualitative insights. Figure [12](https://arxiv.org/html/2507.10473v2#S18.F12 "Figure 12 ‣ 18 Analysis of the learned embedding space ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")(a) presents the distributions of image, time, and location embeddings, appearing in different subspaces. Figures [12](https://arxiv.org/html/2507.10473v2#S18.F12 "Figure 12 ‣ 18 Analysis of the learned embedding space ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")(b) and [12](https://arxiv.org/html/2507.10473v2#S18.F12 "Figure 12 ‣ 18 Analysis of the learned embedding space ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")(c) show more details about the relationship between image and time embeddings. The image embeddings are clustered in the center, surrounded by time embeddings. Notably, the directions of hours and months are well-defined: months are radially distributed, while hours are linearly distributed in a perpendicular direction. For location embeddings (Figures [12](https://arxiv.org/html/2507.10473v2#S18.F12 "Figure 12 ‣ 18 Analysis of the learned embedding space ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")(d-e)), even though the patterns are less pronounced, the embeddings at different latitudes and longitudes still form distinct clusters in the feature space.\
\
![Refer to caption](https://arxiv.org/html/2507.10473v2/pca.png)Figure 12: PCA plots of the embedding spaces in GT-Loc. (a) Distribution of the image, time and location embeddings. (b)-(c) Distribution of the image and time embeddings, colored by the time-of-day and time-of-year respectively. (c)-(d) Distribution of the image and location embeddings, colored by the latitude and longitude respectively.\
\
## 19 Embedding distribution of non-overlapping panorama crops\
\
To further analyze the structure of our learned embedding space, we conduct a qualitative visualization using t-SNE (Figure [13](https://arxiv.org/html/2507.10473v2#S19.F13 "Figure 13 ‣ 19 Embedding distribution of non-overlapping panorama crops ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space")). We begin by randomly selecting 20 panoramas from the CVUSA dataset \[ [42](https://arxiv.org/html/2507.10473v2#bib.bib42 "")\]. From each panorama, we extract four non-overlapping 90∘90^{\\circ} crops, resulting in a total of 80 image embeddings. Since all four crops from a single panorama share the same capture time and location by definition, they serve as a natural test of spatial and temporal consistency in the embedding space. In the resulting t-SNE plot, we observe that embeddings from the same panorama (shown using the same color) form tight and coherent clusters, suggesting that our model effectively encodes shared contextual information across views from the same scene.\
\
![Refer to caption](https://arxiv.org/html/2507.10473v2/tsne_embeddings.png)Figure 13: t-SNE visualization of image embeddings from non-overlapping 90∘90^{\\circ} crops of 20 panoramas sampled from the CVUSA dataset. Each color represents the four crops from a single panorama, which share the same time and location.\
\
## 20 Scalability of the retrieval approach\
\
In Table [11](https://arxiv.org/html/2507.10473v2#S20.T11 "Table 11 ‣ 20 Scalability of the retrieval approach ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space"), we present a comparison of the memory usage and computational cost (measured in FLOPs) between classification and regression baselines and our retrieval-based approach, across gallery sizes ranging from 4,000 to 500,000 samples. At the upper end of this range (500k), we observe that memory usage increases by approximately 950 MB. However, the additional computational cost introduced by the retrieval operations remains minimal, with only a +3.32% increase in FLOPs. It is important to note that the gallery embeddings used for retrieval are precomputed offline in a single forward pass (requiring 9.47 TFLOPs) and are therefore not included in the runtime FLOPs reported in the table.\
\
Table 11: Memory usage and compute cost.\
\
| Method | Memory (GB) | TFLOPs |\
| (gallery size) | Params | Gallery | Total | Forward | Retrieval | Total |\
| CLIP + cls | 1.63 | – | 1.63 | 159.41 | – | 159.41 |\
| CLIP + reg | 1.63 | – | 1.63 | 159.41 | – | 159.41 |\
| GT-Loc (4k) | 1.67 | 0.01 | 1.68 | 159.41 | 0.004 | 159.41 |\
| GT-Loc (20k) | 1.67 | 0.04 | 1.71 | 159.41 | 0.021 | 159.43 |\
| GT-Loc (100k) | 1.67 | 0.19 | 1.86 | 159.41 | 0.105 | 159.52 |\
| GT-Loc (500k) | 1.67 | 0.95 | 2.62 | 159.41 | 0.524 | 159.94 |\
\
## 21 Limitations and reproducibility challenges of existing time prediction methods\
\
Most previous time prediction methods suffer from a lack of standardization in their training and evaluation protocols. For instance, [Zhai et al. \[45\]](https://arxiv.org/html/2507.10473v2#bib.bib45 "") use subsets of the AMOS \[ [8](https://arxiv.org/html/2507.10473v2#bib.bib8 ""), [9](https://arxiv.org/html/2507.10473v2#bib.bib9 "")\] and YFCC100M \[ [34](https://arxiv.org/html/2507.10473v2#bib.bib34 "")\] datasets, without providing the source code or exact dataset splits necessary to replicate their experiments. Additionally, their time prediction evaluation relies on cumulative error plots, but they do not provide a single numerical value summarizing the performance of their model. Similarly, while [Salem et al. \[26\]](https://arxiv.org/html/2507.10473v2#bib.bib26 "") and [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 "") offer datasets and code, they do not include the cross-camera split for zero-shot time prediction—a more challenging and informative evaluation protocol that we adopt. Moreover, their results are presented only qualitatively, though they can be adapted for obtaining quantitative results. [Salem et al. \[27\]](https://arxiv.org/html/2507.10473v2#bib.bib27 "") also omit critical details such as dataset splits for the SkyFinder dataset, do not clarify whether their results corresponds to same- or cross-camera evaluation, and fail to provide the source code for replication. Their use of top-kk accuracy as an evaluation metric further complicates direct comparisons. Notably, none of these methods, with the exception of [Salem et al. \[26\]](https://arxiv.org/html/2507.10473v2#bib.bib26 "") and [Padilha et al. \[22\]](https://arxiv.org/html/2507.10473v2#bib.bib22 ""), compare their time prediction performances against each other, and even these comparisons are only qualitative. Other time prediction approaches, including those by [Tsai et al. \[36\]](https://arxiv.org/html/2507.10473v2#bib.bib36 ""), [Li et al. \[16\]](https://arxiv.org/html/2507.10473v2#bib.bib16 ""), [Lalonde et al. \[14\]](https://arxiv.org/html/2507.10473v2#bib.bib14 ""), and [Hold-Geoffroy et al. \[6\]](https://arxiv.org/html/2507.10473v2#bib.bib6 ""), also face similar challenges, such as a lack of available source code, missing datasets, or datasets that are no longer hosted online. This lack of standardization prevents consistent benchmarking across different methods.\
\
## 22 Qualitative time-of-day results near sunrise and sunset\
\
Predicting the time of day during periods close to sunrise and sunset is particularly challenging due to the visual similarity of scenes captured around these times. The task is further complicated by its strong dependence on geographic location (latitude) and the time of year (month), both of which have a direct influence on the time when these events occur. To illustrate these challenges, Figure [14](https://arxiv.org/html/2507.10473v2#S22.F14 "Figure 14 ‣ 22 Qualitative time-of-day results near sunrise and sunset ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") presents two qualitative examples where GT-Loc makes accurate predictions, as well as two examples where it fails.\
\
![Refer to caption](https://arxiv.org/html/2507.10473v2/sunrise_sunset.png)Figure 14: Qualitative time-of-day results near sunrise and sunset. Top row: correct predictions where the prediction (P) is within one hour of the ground truth (GT). Bottom row: failure cases.\
\
## 23 GT-Loc predictions across different latitudes\
\
Figure [15](https://arxiv.org/html/2507.10473v2#S23.F15 "Figure 15 ‣ 23 GT-Loc predictions across different latitudes ‣ GT-Loc: Unifying When and Where in Images Through a Joint Embedding Space") compares time prediction examples from GT-Loc and TimeLoc, a baseline model trained solely with the visual (𝒱\\mathcal{V}) and temporal (𝒯\\mathcal{T}) encoders. The results suggest that TimeLoc struggles more with hour predictions at higher latitudes (40° to 70°) compared to GT-Loc. In contrast, at moderate latitudes (-40° to 40°), both models exhibit more consistent hour prediction errors, though GT-Loc demonstrates superior performance in month prediction.\
\
![Refer to caption](https://arxiv.org/html/2507.10473v2/qual_gtclip_vs_tclip.png)Figure 15: Sample predictions where GT-Loc outperforms the TimeLoc baseline across different latitudes.