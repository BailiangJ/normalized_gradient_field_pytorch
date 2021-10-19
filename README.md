# Normalized-Gradient-Fields-pytorch
This repo is a pytorch implementation of registration metric Normalized Gradient Field, which can apply to 2D/3D images.

```
@inproceedings{haber2006intensity,
  title={Intensity gradient based registration and fusion of multi-modal images},
  author={Haber, Eldad and Modersitzki, Jan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={726--733},
  year={2006},
  organization={Springer}
} 
@inproceedings{hager2020variable,
  title={Variable Fraunhofer MEVIS RegLib Comprehensively Applied to Learn2Reg Challenge},
  author={H{\"a}ger, Stephanie and Heldmann, Stefan and Hering, Alessa and Kuckertz, Sven and Lange, Annkristin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={74--79},
  year={2020},
  organization={Springer}
}
```
Adopted from repos:

[pytorch_similarity](https://github.com/yuta-hi/pytorch_similarity/blob/master/torch_similarity/modules/gradient_correlation.py)

[ptVreg](https://github.com/visva89/pTVreg/blob/master/mutils/My/image_metrics/metric_ngf.m)

[FAIR.m](https://github.com/C4IR/FAIR.m/blob/master/kernel/distances/NGFdot.m)

[airlab](https://github.com/airlab-unibas/airlab/blob/master/airlab/loss/pairwise.py)
