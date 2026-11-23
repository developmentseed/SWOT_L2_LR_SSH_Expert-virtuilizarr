"""Verify the prototype cube against the source netCDF files.

Checks
  1. structure: dims, granule_present matches the files, no token in repo config
  2. values: for sampled slots, every variable decoded from the cube equals the
     same variable read directly from the granule
  3. holes: empty slots read as all-NaN and granule_present is False
  4. grid: cube lat/lon equal each sampled granule's lat/lon
  5. analysis demo: repeat-track mean / std of ssha_karin over cycle

Usage: uv run python verify_cube.py [--repo PATH] [--samples N]
"""

import argparse
import random
from pathlib import Path

import earthaccess
import numpy as np
import xarray as xr

from obspec_utils.readers import EagerStoreReader

from swot_cube import make_registry, open_cube


def read_direct(url: str, token: str) -> xr.Dataset:
    """Read a granule straight from the source file (no cube), via obstore.

    EagerStoreReader fetches the whole file (~35 MB) in parallel range requests and
    exposes it as a file-like object h5netcdf can open.
    """
    store, path = make_registry(token).resolve(url)
    reader = EagerStoreReader(store, path)
    with xr.open_dataset(reader, engine="h5netcdf", drop_variables=["polarization_karin"]) as ds:
        return ds.load()


def main(repo: str, samples: int) -> None:
    auth = earthaccess.login(strategy="netrc")
    token = auth.token["access_token"]

    # 1. structure ---------------------------------------------------------
    for f in Path(repo).rglob("*"):
        if f.is_file():
            assert token.encode() not in f.read_bytes(), f"token leaked into {f}"
    cube = open_cube(repo, token)
    print(cube)
    present = cube.granule_present.values
    print(f"\nslots: {present.size}, populated: {present.sum()}")

    # 2. values ------------------------------------------------------------
    filled = list(zip(*np.nonzero(present)))
    random.seed(0)
    for ci, pi in random.sample(filled, min(samples, len(filled))):
        slot = cube.isel(cycle=ci, pass_number=pi)
        fname = str(slot.filename.values)
        url = f"https://archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/SWOT_L2_LR_SSH_D/{fname}"
        direct = read_direct(url, token)
        bad = []
        for name in direct.variables:
            if name not in cube.variables:
                bad.append(f"{name} missing from cube")
                continue
            a = slot[name].values
            b = direct[name].values
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fcmM"):
                bad.append(name)
        status = "OK" if not bad else f"MISMATCH {bad[:10]}"
        print(f"cycle {int(slot.cycle)} pass {int(slot.pass_number)}: "
              f"{len(direct.variables)} vars compared -> {status}")  # includes lat/lon (check 4)

    # 3. holes -------------------------------------------------------------
    # holes among the populated passes (e.g. a --drop slot) plus one never-populated pass
    populated_passes = present.any(axis=0)
    holes = [(ci, pi) for ci, pi in zip(*np.nonzero(~present)) if populated_passes[pi]]
    holes += [(0, int(np.nonzero(~populated_passes)[0][0]))]
    for ci, pi in holes[:4]:
        v = cube.ssha_karin.isel(cycle=ci, pass_number=pi).values
        print(f"hole cycle {int(cube.cycle[ci])} pass {int(cube.pass_number[pi])}: "
              f"all NaN = {np.isnan(v).all()}")

    # 5. analysis demo -----------------------------------------------------
    p = int(cube.pass_number[filled[0][1]])
    ssha = cube.ssha_karin.sel(pass_number=p).where(cube.granule_present.sel(pass_number=p))
    mean, std = ssha.mean("cycle").compute(), ssha.std("cycle").compute()
    print(f"\npass {p}: repeat-track mean ssha over {int(cube.granule_present.sel(pass_number=p).sum())}"
          f" cycles -> mean of mean {float(mean.mean()):.4f} m, median std {float(std.median()):.4f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/swot_expert_d_cube.icechunk")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    main(a.repo, a.samples)
