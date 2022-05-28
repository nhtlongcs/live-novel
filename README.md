<div align="center">

# Live Novel
---

**Self-host application can generate illustration from a novel by highlighting certain sentences**

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

</div>

---
## About 
Introducing a new app that is available soon on [nhtlongcs.com](nhtlongcs.com/live-novel). This app can generate images from a novel by highlighting certain sentences. The app is easy to use; all you have to do is enter the name of the book and the highlighted sentences will be displayed as images. This can be a great way for authors to promote their work, or for readers to get imagination of what the scene is about without having to do it themselves.


## Installation 

By prebuilt docker image, you can run the app without any specific configuration.
```bash
$ docker pull nhtlongcs/live-novel
$ docker run --rm --name live-novel --gpus device=0 -p 5001:5001 -it -v $(pwd)/:/home/dreamer/workspace/src/ live-novel:latest /bin/bash
```
<details>
<summary>Other installations</summary>

To install **Live-Novel** and customize locally
<!-- ssh -N -f -p 12156 -L localhost:8080:localhost:8080 root@ssh4.vast.ai -->
```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t live-novel:latest .
$ docker run --rm --name live-novel --gpus device=0 -p 5001:5001 -it -v $(pwd)/:/home/dreamer/workspace/src/ live-novel:latest /bin/bash
```
</details>


## Acknoledgements

To make this app work, we heavily adopted from [disco diffusion repo](https://github.com/alembics/disco-diffusion) and related projects (e.g. [gui-diffusion](https://github.com/crowsonkb/guided-diffusion), [resize-right](https://github.com/assafshocher/ResizeRight), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [taming transformers](https://github.com/CompVis/taming-transformers))

To enhance the result quality, we used [ESRGAN](https://github.com/xinntao/ESRGAN) for image super resolution.
