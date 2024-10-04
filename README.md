# Variational Diffusion Posterior Sampling With Midpoint Guidance

[**Link to our Paper**]()

The code of MGPS algorithm for solving Bayesian inverse problems with Diffusion Models as a prior.
The algorithm handles linear and non-linear problems including those with Latent Diffusion Models (LDM) priors. 

<div align="center">
  <img src="material/all-datasets.png" />
</div>

- 1st and 2nd row: FFHQ
- 3rd and 4th row: Imagenet
- 5th and 6th row: LDM on FFHQ 

See the appendix of our paper for more examples.


## Code installation

### Install project dependencies

Install the code in editable mode

```bash
pip install -e .
```

This command will also download the code dependencies.
Further details about dependencies are in ``setup.py``.

For convenience, the code of these repositories were moved inside ``src`` folder to avoid installation conflicts.

- https://github.com/bahjat-kawar/ddrm
- https://github.com/openai/guided-diffusion
- https://github.com/NVlabs/RED-diff
- https://github.com/mlomnitz/DiffJPEG
- https://github.com/CompVis/latent-diffusion


### Set configuration paths

Since we use the project path for cross referencing, namely open configuration files, ensure to define it in ``src/local_paths.py`` (copy/paste the output of ``pwd`` command)

After [downloading](#downloading-checkpoints) the models checkpoints, make sure to put the corresponding paths in the configuration files

- Model checkoints
  - ``configs/ffhq_model.yaml``
  - ``configs/imagenet_model.yaml``
  - ``configs/ffhq-ldm-vq-4.yaml``
- Nonlinear blur
  - ``src/nonlinear_blurring/option_generate_blur_default.yml``


## Assets

We provide few images of FFHQ and Imagenet.
Some of the degradation operator are also provided as checkpoints to alleviate the initialization overhead.

These are located in ``assets/`` folder

```
  assets/
  ├── images/
  ├──── ffhq/
  |       └── im1.png
  |       └── ...
  ├──── imagenet/
  |       └── im1.png
  |       └── ...
  ├── operators/
  |    └── outpainting_half.pt
  |    └── ...
```


## Reproduce experiments

We provide two scripts, ``test_images.py`` and ``test_gaussian.py`` to run the experiments.

### Image restoration tasks

In addition to our algorithm, several state-of-the-art algorithms are supported
``"mgps"`` (ours), ``"diffpir"``, ``"ddrm"``, ``"ddnm"``, ``"dps"``, ``"pgdm"``, ``"psld"``, ``"reddiff"``, ``"resample"``.

their hyperparameters are defined in ``configs/experiments/sampler/`` folder.

we also support several imaging tasks

- **Inpainting**: ``"inpainting_center"``, ``"outpainting_half"``, ``"outpainting_top"``
- **Blurring**: ``"blur"``,  ``"blur_svd"`` (SVD version of blur), ``"motion_blur"``,  ``"nonlinear_blur"``,
- **JPEG dequantization**:  ``"jpeg{QUALITY}"`` (Quality is an integer in [1, 99], example ``"jpeg2"``)
- **Super Resolution**: ``"sr4"``, ``"sr16"``
- **Others**: ``"phase_retrieval"``, ``"high_dynamic_range"``

To run an experiment, execute the following command

```bash
python test_images.py task=inpainting_center sampler=mgps dataset=ffhq device=cuda:0
```

### Gaussian case with midpoint guidance

Use the script ``test_gaussian.py`` to run the experiment of Gaussian case with midpoint guidance as described in Example 3.2.


## Downloading checkpoints

- [Imagnet](https://github.com/openai/guided-diffusion)
- [FFHQ](https://github.com/DPS2022/diffusion-posterior-sampling)
- FFHQ LDM: [denoiser](https://ommer-lab.com/files/latent-diffusion/ffhq.zip), [autoencoder](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip)
- [Nonlinear blur operator](https://drive.google.com/file/d/1xUvRmusWa0PaFej1Kxu11Te33v0JvEeL/view?usp=drive_link)
