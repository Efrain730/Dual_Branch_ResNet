## Dual-Branch Residual Network for Pulmonary Textures Classification

Our method is utilized to classify seven categories of pulmonary textures of diffuse lung diseases, which is shown as follows:
![avatar](/fig/texture_example.png)

We proposed a dual-branch residual network to achieve this mission, the architecture is shown as follows:
![avatar](/fig/network_structure.png)

The inputs of network are composed of two groups. One group are image patches extracted from original CT volumes, and another one are eigen-values derived from Hessian matrices, which are calculated on each pixel of corresponding image patches. Hessian matrix is defined as follows:
![avatar](/fig/hessian_matrix.png)

## Ablation Study

We alter network structure slightly, the ablation study experimental results are shown as follows:
![avatar](/fig/ablation.png)

## Comparison with the State-of-the-Art

Comparisons with other state-of-the-art methods are exhibited as follows:
![avatar](/fig/SoA.png)
