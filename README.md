# Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization [NeurIPS 2023]

[Jameel Hassan *](https://jameelhassan.github.io/), [Hanan Gani *](https://hananshafi.github.io/), [Noor Hussein](https://ae.linkedin.com/in/noor-hussein-67566a183), [Uzair Khattak](https://muzairkhattak.github.io/), [Muzammal Naseer](https://muzammal-naseer.com/), [Fahad Khan](https://sites.google.com/view/fahadkhans/home),  [Salman Khan](https://salman-h-khan.github.io/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.01459)
[![Poster](https://img.shields.io/badge/Poster-PDF-8333FF)](https://jameelhassan.github.io/promptalign/static/images/poster.png)
[![Slides](https://img.shields.io/badge/Slides-PDF-87CEEB)](https://github.com/jameelhassan/PromptAlign/blob/master/docs/slides.pdf) 
[![Video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://recorder-v3.slideslive.com/#/share?share=89606&s=2a435db5-fe74-47ff-91fc-f904a679f44d)

Official implementation of the paper "Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization".

<hr>

## Contents

1. [Updates](#News)
2. [Highlights](#Highlights)
3. [Main Contributions](#Main-Contributions)
4. [Installation](#Installation)
5. [Datasets](#Datasets)
6. [Run PromptAlign](#Run-PromptAlign)
7. [Results](#Results)
8. [Citation](#Citation)
9. [Contact](#Contact)
10. [Acknowledgements](#Acknowledgements)

<hr>

## Updates

* Code for PrompAlign is released. [November 3, 2023]
* Our paper is accepted at NeurIPS 2023 [September 22, 2023]

## Highlights
![concept-diagram](https://jameelhassan.github.io/promptalign/static/images/conceptdiagram.png)

> **Abstract:** *The promising zero-shot generalization of vision-language models such as CLIP
has led to their adoption using prompt learning for numerous downstream tasks.
Previous works have shown test-time prompt tuning using entropy minimization
to adapt text prompts for unseen domains. While effective, this overlooks the key
cause for performance degradation to unseen domains – distribution shift. In this
work, we explicitly handle this problem by aligning the out-of-distribution (OOD)
test sample statistics to those of the source data using prompt tuning. We use a
single test sample to adapt multi-modal prompts at test time by minimizing the
feature distribution shift to bridge the gap in the test domain. Evaluating against the
domain generalization benchmark, our method improves zero-shot top-1 accuracy
beyond existing prompt-learning techniques, with a 3.08% improvement over the
baseline MaPLe. In cross-dataset generalization with unseen categories across 10
datasets, our method improves by 1.82% compared to the existing state-of-the-art.*
>
<hr>

## Main Contributions
* **Distribution alignment using a single sample:** Given only a single test sample, we introduce a distribution alignment strategy for V-L
models to improve test-time adaptation using lightweight prompt learning strategy.
* **Offline statistics for distribution alignment:** The proposed strategy formulates a distribution alignment loss that utilizes offline computed
source data statistics to encourage the test sample token distributions to be aligned with the source data token distributions. This harmonically combines
token distribution alignment with entropy minimization using multi-modal prompt learning.
* **Proxy Source dataset:** Since CLIP-pre-training data is not publicly released, we study the statistics of ImageNet
as a possible candidate for the source distribution, and our empirical results show that
ImageNet is an effective proxy source dataset for large-scale V-L models such as CLIP.


## Installation
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](https://github.com/jameelhassan/PromptAlign/blob/master/docs/INSTALL.md)


## Data Preparation
Please follow the instructions at [DATASETS.md](https://github.com/jameelhassan/PromptAlign/blob/master/docs/datasets.md) to prepare all datasets.

## Run PromptAlign
Please refer to the [RUN.md](https://github.com/jameelhassan/PromptAlign/blob/master/docs/run.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.

### Results

#### Domain Generalization

<div align="center">

| Method           |  IN-V2 | IN-Sketch | IN-A | IN-R  | OOD Average |
|------------------|:----------:|:-----------:|:----------:|:---------------:|:-----------:|
| [CLIP](https://arxiv.org/abs/2103.00020)         | 60.86   |    46.06    |    47.87   |      73.98      |     57.20    |
| [CoOp](https://arxiv.org/abs/2109.01134)            |    64.20   |   47.99    |    49.71   |      75.21      |      59.28    |
| [CoCoOp](https://arxiv.org/abs/2203.05557)         |    64.07   |    48.75    |    50.63   |      76.18      |     59.91    |
| [MaPLe](https://arxiv.org/abs/2210.03117)          |   64.07  |   49.15   |    50.90    |    76.98   |       60.28    |
| [TPT + CLIP](https://arxiv.org/abs/2209.07511) |   64.35   |    47.94    |    54.77   |      77.06      |      60.81    |
| [TPT + CoOp](https://arxiv.org/abs/2209.07511)       |    **66.83**   |    49.29    |    57.95   |      77.27      |     62.84    |
| [TPT + CoCoOp](https://arxiv.org/abs/2209.07511)     |     64.85   |    48.27   |    58.47   |      78.65      |  62.61    |
| TPT + MaPLe     |      64.87   |    48.16    |    58.08   |      78.12      |  62.31    |
| PromptAlign     |      65.29   |    **56.23**    |    **59.37**  |      **79.33**      |  **63.55**    |
</div>


## Citation
If you use our work, please consider citing: 
```
@inproceedings{samadh2023align,
  title={Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization},
  author={Samadh, Jameel Hassan Abdul and Gani, Hanan and Hussein, Noor Hazim and Khattak, Muhammad Uzair and Naseer, Muzammal and Khan, Fahad and Khan, Salman},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Contact
Should you have any questions, please create an issue in this repository or contact at jameel.hassan@mbzuai.ac.ae or hanan.ghani@mbzuai.ac.ae

## Acknowledgements
We thank the authors of [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [TPT](https://github.com/azshue/TPT), and [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp) for their open-source implementation and instructions on data preparation. 
