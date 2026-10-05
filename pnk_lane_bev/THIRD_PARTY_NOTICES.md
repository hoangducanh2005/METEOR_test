# Third-party notice

The UFLD-v2 model wrapper in this package is adapted from
[cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2),
commit `c903880678454dfd9b55a63022368db05c00bc6d`.

Ultra-Fast-Lane-Detection-v2 is distributed under the MIT License, copyright
(c) 2022 Zequn Qin. The upstream license and copyright notice apply to the
adapted model/decode implementation in `src/pnk_lane_bev/ufld_v2.py`.

The CLRNet inference graph in `src/pnk_lane_bev/clrnet.py` is adapted from
[Turoad/CLRNet](https://github.com/Turoad/CLRNet), commit
`7269e9d1c1c650343b6c7febb8e764be538b1aed`. CLRNet is distributed under the
Apache License 2.0. The local adapter replaces MMCV and the compiled CUDA NMS
extension with current PyTorch operations; model parameters and decoding remain
compatible with the official `culane_r18.pth` release.

The line-distance NMS behavior follows the CLRNet NMS source, which includes a
BSD 3-Clause notice for its original implementation. Copies of the applicable
license texts are stored in `licenses/`.
