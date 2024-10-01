# VARIATIONAL DIFFUSION POSTERIOR SAMPLING WITH MIDPOINT GUIDANCE

The code of MGPS algorithm for solving Bayesian inverse problems with Diffusion models as prior.
The algorithm solves linear and non-linear problems.


## Code installation

### Install project dependencies

Install the code in editable mode

```bash
pip install -e .
```

This command will also download the code dependencies.
Further details about dependencies are in ``setup.py``.

For convenience, the code of these repositories were moved inside ``src`` folder

- https://github.com/bahjat-kawar/ddrm
- https://github.com/openai/guided-diffusion
- https://github.com/NVlabs/RED-diff
- https://github.com/mlomnitz/DiffJPEG

to avoid installation conflicts.


### Set configuration paths

TODO
  - path of repo
  - path large files
  - path of models weights


## Large files

The models checkpoints, datasets were ignored as they contain large files.
Make sure to create a folder ``large_files`` and download the right files and folders.

To avoid path conflict, ensure to insert in ``src/local_paths.py`` script

- the absolute path of the repository
- the path of the folder ``large_files``

and update the ``model_path`` in the configuration files ``ffhq_model.yaml`` and ``imagenet_model.yaml``.

The ``large_files`` folder have the following structure.
Make sure to preserve it.

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


## A tour on the repository scripts

TODO
- Provide description on the CLI for images
- Running Experiments of Gaussian and t_mid



## Downloading checkpoints

- [Imagnet](https://github.com/openai/guided-diffusion)
- [FFHQ](https://github.com/DPS2022/diffusion-posterior-sampling)


TODO add link to
- latent FFHQ
- non linear blur


## TODO
- Add images for ffhq and imagenet
- Add degradation operators