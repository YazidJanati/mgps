python test_images.py task=inpainting_center sampler=mgps sampler.nsteps=50 sampler.nsamples=1 dataset=ffhq im_idx="00018" device=cuda:0

# # for imagenet
# python test_images.py task=inpainting_center sampler=mgps sampler.nsteps=50 sampler.nsamples=1 dataset=imagenet im_idx="00007" device=cuda:0

# # for FFHQ with latent diffusion models
# python test_images.py task=inpainting_center sampler=mgps sampler.nsteps=50 sampler.nsamples=1 dataset=ffhq_ldm im_idx="00018" device=cuda:0
