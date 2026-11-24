"""Verify the prototype cube against the source netCDF files.

Checks
  1. structure: dims, granule_present matches the files, no token anywhere in the repo,
     references are relative (resolve to S3 with the stored config, HTTPS with the override)
  2. values: for sampled slots, every variable decoded from the cube equals the
     same variable read directly from the granule
  3. holes: empty slots read as all-NaN and granule_present is False
  4. grid: cube lat/lon equal each sampled granule's lat/lon
  5. analysis demo: repeat-track mean / std of SSHA over cycle
     (ssha_karin_2 + height_cor_xover, ssha_karin_2_qual == 0)

Usage: uv run python verify_cube.py [--repo PATH] [--samples N]
"""

import argparse
import random
from pathlib import Path

import icechunk
import numpy as np
import xarray as xr

from obspec_utils.readers import EagerStoreReader

from swot_cube import HTTPS_PREFIX, S3_PREFIX, login, make_registry, open_cube, open_repo


def read_direct(url: str, auth) -> xr.Dataset:
    """Read a granule straight from the source file (no cube), via obstore.

    EagerStoreReader fetches the whole file (~35 MB) in parallel range requests and
    exposes it as a file-like object h5netcdf can open.
    """
    store, path = make_registry(auth).resolve(url)
    reader = EagerStoreReader(store, path)
    with xr.open_dataset(reader, engine="h5netcdf", drop_variables=["polarization_karin"]) as ds:
        return ds.load()


def main(repo: str, samples: int) -> None:
    auth = login()
    token = auth.token["access_token"]  # only to check it never lands in the repo

    # 1. structure ---------------------------------------------------------
    for f in Path(repo).rglob("*"):
        if f.is_file():
            assert token.encode() not in f.read_bytes(), f"token leaked into {f}"
    stored = icechunk.Repository.open(icechunk.local_filesystem_storage(repo))
    for label, r, prefix in [("stored config", stored, S3_PREFIX),
                             ("https override", open_repo(repo, auth, "https"), HTTPS_PREFIX)]:
        batch = next(r.readonly_session("main").store.array_chunk_iterator("ssha_karin_2", 1))
        loc = batch[2][0]  # resolved location of the first chunk
        assert loc.startswith(prefix), f"{label}: {loc} does not start with {prefix}"
        print(f"relative refs, {label}: {loc}")
    cube = open_cube(repo, auth)
    print(cube)
    present = cube.granule_present.values
    print(f"\nslots: {present.size}, populated: {present.sum()}")

    # 2. values ------------------------------------------------------------
    filled = list(zip(*np.nonzero(present)))
    random.seed(0)
    for ci, pi in random.sample(filled, min(samples, len(filled))):
        slot = cube.isel(cycle=ci, pass_number=pi)
        fname = str(slot.filename.values)
        url = f"{HTTPS_PREFIX}SWOT_L2_LR_SSH_D/{fname}"
        direct = read_direct(url, auth)
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
    sub = cube.sel(pass_number=p)
    ssha = (sub.ssha_karin_2 + sub.height_cor_xover).where(sub.ssha_karin_2_qual == 0)
    mean, std = ssha.mean("cycle").compute(), ssha.std("cycle").compute()
    print(f"\npass {p}: repeat-track mean ssha over {int(cube.granule_present.sel(pass_number=p).sum())}"
          f" cycles -> mean of mean {float(mean.mean()):.4f} m, median std {float(std.median()):.4f} m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="data/swot_expert_d_cube.icechunk")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    main(a.repo, a.samples)
