# DKMPP
This repository contains the implementation for the paper "Integration-free training for spatio-temporal multimodal covariate deep kernel point processes," accepted at NeurIPS 2023. 

We further improved the score matching estimation method for point processes. Please refer to this NeurIPS 2024 paper "Is Score Matching Suitable for Estimating Point Processes?" [[code]](https://github.com/KenCao2007/WSM_TPP)


## Abstract
In this study, we propose a novel deep spatio-temporal point process model, Deep Kernel Mixture Point Processes (DKMPP), that incorporates multimodal covariate information. DKMPP is an enhanced version of Deep Mixture Point Processes (DMPP), which uses a more flexible deep kernel to model complex relationships between events and covariate data, improving the model's expressiveness. To address the intractable training procedure of DKMPP due to the non-integrable deep kernel, we utilize an integration-free method based on score matching, and further improve efficiency by adopting a scalable denoising score matching method. Our experiments demonstrate that DKMPP and its corresponding score-based estimators outperform baseline models, showcasing the advantages of incorporating covariate information, utilizing a deep kernel, and employing score-based estimators.

## Reference
```
@inproceedings{NEURIPS2023_4eb2c0ad,
 author = {Zhang, Yixuan and Kong, Quyu and Zhou, Feng},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {25031--25049},
 publisher = {Curran Associates, Inc.},
 title = {Integration-free Training for Spatio-temporal Multimodal Covariate Deep Kernel Point Processes},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

```
