from setuptools import setup


requirements = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers[torch]",
    "lpips",
    "Pillow",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "omegaconf",
    "tqdm",
    "PyYAML",
    "einops",
    "lightning",
    "taming-transformers-rom1504",
    "hydra-core",
    "pandas",
]


setup(
    name="mgps",
    version="0.0.0",
    install_requires=requirements,
)
