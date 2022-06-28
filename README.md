<div align="center">

# Live Novel
---

**Self-host application can generate illustration from a novel by highlighting certain sentences**

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

</div>

---
## About 
Introducing a new app that is ~now available~ (WIP) on [novel.nhtlongcs.com](https://novel.nhtlongcs.com/). This app can generate images from a novel by highlighting certain sentences. The app is easy to use; all you have to do is enter the name of the book and the highlighted sentences will be displayed as images. This can be a great way for authors to promote their work, or for readers to get imagination of what the scene is about without having to do it themselves.


## Installation 

By prebuilt docker image, you can run the app without any specific configuration.
```bash
$ docker pull nhtlongcs/live-novel
```
<details>
<summary>Other installations</summary>

To install **Live-Novel** and customize locally
<!-- ssh -N -f -p 12156 -L 0.0.0.0:8080:0.0.0.0:8080 root@ssh4.vast.ai -->
```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t live-novel:latest .
```
</details>

## Usage 

```bash
$ docker run --rm --name live-novel --gpus device=0 -p 5001:5001 -it live-novel:latest /bin/bash
$ cd ~/workspace 
$ python app.py --port XXXXX 
```

Use environment variable `CUDA_VISIBLE_DEVICES=-1` to disable cuda.

For debug mode, server will respone a random image. To enable debug mode, use flag `--debug` when running `app.py`

For customize purpose, please clone this repo and use flag `-v $(pwd)/:/home/dreamer/workspace/src/` to replace source code in the docker

## Acknoledgements

To make this app work, we heavily adopted from [disco diffusion repo](https://github.com/alembics/disco-diffusion) and related projects (e.g. [gui-diffusion](https://github.com/crowsonkb/guided-diffusion), [resize-right](https://github.com/assafshocher/ResizeRight), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [taming transformers](https://github.com/CompVis/taming-transformers))

To enhance the result quality, we used [ESRGAN](https://github.com/xinntao/ESRGAN) for image super resolution.
