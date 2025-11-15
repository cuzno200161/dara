![DARA logo](https://github.com/idocx/dara/blob/main/logo/dara.jpg?raw=true)

# DARA: Data-driven automated Rietveld analysis for phase search and refinement
[![GitHub licence](https://img.shields.io/github/license/idocx/dara)](https://github.com/idocx/dara/blob/main/LICENSE)
[![arXiv:2510.19667](https://img.shields.io/badge/arXiv-2510.19667-b31b1b?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2510.19667)
[![Pytest](https://github.com/idocx/dara/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/idocx/dara/actions/workflows/pytest.yaml)
[![Ruff Check](https://github.com/idocx/dara/actions/workflows/ruff.yaml/badge.svg?branch=main)](https://github.com/idocx/dara/actions/workflows/ruff.yaml)

Automated phase search with BGMN.
## Installation
```bash
pip install dara-xrd
```

For more details about installation, please refer to [installation guide](https://idocx.github.io/dara/install.html).

## Web Server
Dara ships with a browser-based web server for an out-of-box experience of Dara. To launch the webserver, run
```bash
dara server
```

Then you can open http://localhost:8898 to see an application that can submit, manage, and view jobs.


## Documentation
For more details about usage, please refer to the [documentation](https://idocx.github.io/dara/).

## Citation
If you use DARA in your research, please consider citing the following paper:

```
@article{fei2025dara,
  title={Dara: Automated multiple-hypothesis phase identification and refinement from powder X-ray diffraction},
  author={Fei, Yuxing and McDermott, Matthew J and Rom, Christopher L and Wang, Shilong and Ceder, Gerbrand},
  journal={arXiv preprint arXiv:2510.19667},
  year={2025}
}
```