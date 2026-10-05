"""Download official UFLD-v2 and CLRNet CULane ResNet-18 checkpoints."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import gdown


model_dir = Path(__file__).resolve().parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


ufld = model_dir / "culane_res18.pth"
if ufld.exists() and digest(ufld) == "956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967":
    print(f"UFLD-v2 checkpoint already exists: {ufld}")
else:
    result = gdown.download(id="1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W", output=str(ufld), quiet=False)
    if not result or not ufld.exists():
        raise RuntimeError("UFLD-v2 checkpoint download failed")
    if digest(ufld) != "956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967":
        raise RuntimeError("UFLD-v2 checkpoint checksum mismatch")
    print(f"Downloaded UFLD-v2 checkpoint: {ufld}")

clrnet = model_dir / "culane_r18.pth"
if clrnet.exists() and digest(clrnet) == "042f0a1e8d5f1cdfcf97b21b93c6f40d3d67cdeb682a5fcf80e49e65af887322":
    print(f"CLRNet checkpoint already exists: {clrnet}")
else:
    archive = model_dir / "culane_r18.pth.zip"
    urllib.request.urlretrieve(
        "https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip",
        archive,
    )
    with zipfile.ZipFile(archive) as payload:
        payload.extract("culane_r18.pth", model_dir)
    if digest(clrnet) != "042f0a1e8d5f1cdfcf97b21b93c6f40d3d67cdeb682a5fcf80e49e65af887322":
        raise RuntimeError("CLRNet checkpoint checksum mismatch")
    print(f"Downloaded CLRNet checkpoint: {clrnet}")
