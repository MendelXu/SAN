FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN pip install cython scipy shapely timm h5py submitit scikit-image wandb setuptools numpy Pillow pycocotools~=2.0.4 fvcore tabulate tqdm ftfy regex opencv-python open_clip_torch cityscapesscripts tensorboard
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install opencv-python-headless==4.5.5.64
