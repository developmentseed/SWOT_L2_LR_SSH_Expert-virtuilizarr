# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["earthaccess", "obstore", "obspec-utils", "xarray", "h5netcdf", "h5py", "numpy", "pandas"]
# ///
"""Is the SWOT L2 LR SSH Expert (version D) grid fixed cycle to cycle?

For a few science-phase passes, read latitude/longitude (and nadir) from ~10
cycles and measure how far each pixel moves relative to a reference cycle.
If displacements are a small fraction of the 2 km pixel spacing, line i /
pixel j means the same place every cycle and a (cycle, pass) cube supports
direct reductions over `cycle`.

Usage: uv run check_grid_stability.py [PASS ...]
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from obspec_utils.readers import EagerStoreReader
from obstore.store import HTTPStore

SHORT_NAME = "SWOT_L2_LR_SSH_EXPERT_D"
SCIENCE_START = "2023-07-26"  # cycle 1 of the 21-day science orbit (Cal/Val excluded)
N_CYCLES = 10
PASSES = [int(p) for p in sys.argv[1:]] or [13, 300]
VARS = ["latitude", "longitude", "latitude_nadir", "longitude_nadir"]
NAME_RE = re.compile(r"Expert_(\d{3})_(\d{3})_(\d{8}T\d{6})_\d{8}T\d{6}_(\w{4})_(\d{2})\.nc")


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def pick_granules(pass_num):
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        granule_name=f"SWOT_L2_LR_SSH_Expert_*_{pass_num:03d}_*.nc",
        temporal=(SCIENCE_START, pd.Timestamp.now().strftime("%Y-%m-%d")),
    )
    rows = []
    for g in results:
        url = g.data_links(access="external")[0]
        m = NAME_RE.search(url)
        if m:
            rows.append(dict(cycle=int(m[1]), crid=m[4], counter=int(m[5]), url=url))
    df = pd.DataFrame(rows)
    # one granule per cycle: latest product counter, prefer PGD (reprocessed) over PID
    df["crid_rank"] = df.crid.str.startswith("PGD").astype(int)
    df = (df.sort_values(["cycle", "crid_rank", "counter"])
            .groupby("cycle").tail(1).sort_values("cycle"))
    idx = np.unique(np.linspace(0, len(df) - 1, N_CYCLES).round().astype(int))
    return df.iloc[idx]


HTTPS_HOST = "https://archive.swot.podaac.earthdata.nasa.gov/"
_store: HTTPStore | None = None  # set in __main__ once logged in


def read_coords(url):
    """Fetch the whole file (~35 MB) via obstore in parallel ranges; read the coords."""
    reader = EagerStoreReader(_store, url.removeprefix(HTTPS_HOST))
    with xr.open_dataset(reader, engine="h5netcdf") as ds:
        return {v: ds[v].values for v in VARS}


def check_pass(pass_num):
    sel = pick_granules(pass_num)
    print(f"\n=== pass {pass_num:03d}: cycles {sel.cycle.tolist()} "
          f"(crids {sorted(set(sel.crid))}) ===")
    with ThreadPoolExecutor(5) as ex:
        coords = list(ex.map(read_coords, sel.url))

    shapes = {c["latitude"].shape for c in coords}
    print("swath shapes:", shapes)
    if len(shapes) > 1:
        print("  -> shapes differ across cycles; line/pixel index cannot be fixed grid")
        return

    ref = coords[0]
    rows = []
    for cyc, c in zip(sel.cycle, coords):
        d = haversine_m(ref["latitude"], ref["longitude"], c["latitude"], c["longitude"])
        dn = haversine_m(ref["latitude_nadir"], ref["longitude_nadir"],
                         c["latitude_nadir"], c["longitude_nadir"])
        valid = np.isfinite(d)
        rows.append(dict(
            cycle=cyc,
            valid_frac=valid.mean().round(3),
            swath_median_m=np.nanmedian(d).round(1),
            swath_p99_m=np.nanpercentile(d, 99).round(1),
            swath_max_m=np.nanmax(d).round(1),
            nadir_median_m=np.nanmedian(dn).round(1),
            nadir_max_m=np.nanmax(dn).round(1),
        ))
    print(f"displacement vs cycle {sel.cycle.iloc[0]} (pixel spacing = 2000 m):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    token = earthaccess.login(strategy="netrc").token["access_token"]
    _store = HTTPStore.from_url(
        HTTPS_HOST, client_options={"default_headers": {"Authorization": f"Bearer {token}"}}
    )
    for p in PASSES:
        check_pass(p)
