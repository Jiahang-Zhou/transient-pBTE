# PINN for transient phonon BTE
[Physics-informed neural networks for solving time-dependent mode-resolved phonon Boltzmann transport equation](https://www.nature.com/articles/s41524-023-01165-7)

[Jiahang Zhou](https://scholar.google.com/citations?user=Vuy41rUAAAAJ&hl=en), [Ruiyang Li](https://scholar.google.com/citations?hl=en&user=Kezfhw8AAAAJ), [Tengfei Luo](https://scholar.google.com/citations?hl=en&user=VIiy6ugAAAAJ)

## Abstract
The phonon Boltzmann transport equation (BTE) is a powerful tool for modeling and understanding micro-/nanoscale thermal transport in solids, where Fourier’s law can fail due to non-diffusive effect when the characteristic length/time is comparable to the phonon mean free path/relaxation time. However, numerically solving phonon BTE can be computationally costly due to its high dimensionality, especially when considering mode-resolved phonon properties and time dependency. In this work, we demonstrate the effectiveness of physics-informed neural networks (PINNs) in solving time-dependent mode-resolved phonon BTE. The PINNs are trained by minimizing the residual of the governing equations, and boundary/initial conditions to predict phonon energy distributions, without the need for any labeled training data. The results obtained using the PINN framework demonstrate excellent agreement with analytical and numerical solutions. Moreover, after offline training, the PINNs can be utilized for online evaluation of transient heat conduction, providing instantaneous results, such as temperature distribution. It is worth noting that the training can be carried out in a parametric setting, allowing the trained model to predict phonon transport in arbitrary values in the parameter space, such as the characteristic length. This efficient and accurate method makes it a promising tool for practical applications such as the thermal management design of microelectronics.

## Citation

If you find this method useful for your research, we kindly request that you acknowledge it by citing:
```latex
@article{zhou_physics-informed_2023,
	title = {Physics-informed neural networks for solving time-dependent mode-resolved phonon Boltzmann transport equation},
	volume = {9},
	issn = {2057-3960},
	url = {https://doi.org/10.1038/s41524-023-01165-7},
	doi = {10.1038/s41524-023-01165-7},
	abstract = {The phonon Boltzmann transport equation ({BTE}) is a powerful tool for modeling and understanding micro-/nanoscale thermal transport in solids, where Fourier’s law can fail due to non-diffusive effect when the characteristic length/time is comparable to the phonon mean free path/relaxation time. However, numerically solving phonon {BTE} can be computationally costly due to its high dimensionality, especially when considering mode-resolved phonon properties and time dependency. In this work, we demonstrate the effectiveness of physics-informed neural networks ({PINNs}) in solving time-dependent mode-resolved phonon {BTE}. The {PINNs} are trained by minimizing the residual of the governing equations, and boundary/initial conditions to predict phonon energy distributions, without the need for any labeled training data. The results obtained using the {PINN} framework demonstrate excellent agreement with analytical and numerical solutions. Moreover, after offline training, the {PINNs} can be utilized for online evaluation of transient heat conduction, providing instantaneous results, such as temperature distribution. It is worth noting that the training can be carried out in a parametric setting, allowing the trained model to predict phonon transport in arbitrary values in the parameter space, such as the characteristic length. This efficient and accurate method makes it a promising tool for practical applications such as the thermal management design of microelectronics.},
	pages = {212},
	number = {1},
	journaltitle = {npj Computational Materials},
	author = {Zhou, Jiahang and Li, Ruiyang and Luo, Tengfei},
	date = {2023-11-25},
}
```

