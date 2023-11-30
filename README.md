<h1 align="center">
  🐍 PRIOR
</h1>


<h3 align="center"><em>A Python Package for Seamless Data Distribution in AI Workflows</em></h3>

![DALL·E 2022-09-12 18 02 32 - A friendly green snake typing on a computer on the floor](https://user-images.githubusercontent.com/28768645/189784788-22986a02-d56e-4937-8c8e-e58685e8b72d.png)


<div align="center">
  <a href="https://zenodo.org/badge/latestdoi/497726192">
    <img src="https://zenodo.org/badge/497726192.svg" alt="DOI" />
  </a>
  <a href="//github.com/allenai/prior/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/allenai/prior.svg?color=blue">
  </a>
  <a href="//github.com/allenai/prior/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/prior.svg">
  </a>
  <a href="//pepy.tech/project/prior" target="_blank">
    <img alt="Downloads" src="https://pepy.tech/badge/prior">
  </a>
</div>



## Installation

Install the `prior` package with pip:

```bash
pip install prior
```

You'll also need [git](https://git-scm.com/) and the [git-lfs](https://git-lfs.com/) extension installed.

## Datasets

- ProcTHOR-10k [[GitHub]](https://github.com/allenai/procthor-10k)

```python
import prior
prior.load_dataset("procthor-10k")
```

- Object Nav Evaluation [[GitHub]](https://github.com/allenai/object-nav-eval)

```python
import prior
prior.load_dataset("object-nav-eval")
```

## Models

- ProcTHOR Models [[GitHub]](https://github.com/allenai/procthor-models)

```python
import prior
prior.load_model(project="procthor-models", model="object-nav-pretraining")
```

## Example Usage

To use a public Python dataset, simply run:

```python
import prior
dataset = prior.load_dataset("test-dataset", entity="mattdeitke", revision="main")
```

Here, `revision` can be either a tag, branch, or commit hash.

## Private Datasets

If you want to use a private dataset, make sure you're either:

1. Already logged into GitHub from the command line, and able to pull a private repo.
2. Set the GITHUB_TOKEN environment variable to a GitHub authentication token with read access to private repositories (e.g., `export GITHUB_TOKEN=<token>`). You can generate a GitHub authentication token [here](https://github.com/settings/tokens).
3. Set the `gh_auth_token` global variable in the `prior` package with:

```python
import prior
prior.gh_auth_token = "<token>"
```

## Citation

To cite the PRIOR package, please use:

```bibtex
@software{prior,
  author={Matt Deitke and Aniruddha Kembhavi and Luca Weihs},
  doi={10.5281/zenodo.7072830},
  title={{PRIOR: A Python Package for Seamless Data Distribution in AI Workflows}},
  url={https://github.com/allenai/prior},
  year={2022}
}
```
