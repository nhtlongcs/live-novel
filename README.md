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


 ### Misuse, Malicious Use, and Out-of-Scope Use
_Note: This section is taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), but applies in the same way to Stable Diffusion v1_.

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

#### Misuse and Malicious Use
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.


## Comments 

- My codebase for the application builds heavily on [Jina's DiscoArt](https://github.com/jina-ai/discoart). And use [Stable Diffusion Model](https://github.com/CompVis/stable-diffusion) as the core generation model.
Thanks for open-sourcing!

## Citation

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
```
