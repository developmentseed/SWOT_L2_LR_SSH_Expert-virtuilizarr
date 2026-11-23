"""Prototype: SWOT L2 LR SSH Expert (version D) as a virtual (cycle, pass_number) cube.

Layout
------
- Swath variables:  (cycle, pass_number, num_lines, num_pixels)
- Line variables:   (cycle, pass_number, num_lines)            e.g. time, *_nadir
- Side variables:   (cycle, pass_number, num_lines, num_sides)
- Swath grid:       latitude/longitude (pass_number, num_lines, num_pixels)
                    -- fixed per pass across cycles (see check_grid_stability.py)
- Granule coords:   (cycle, pass_number) granule_present, filename, crid, times, bbox, ...

Each (cycle, pass_number) slot holds one granule; every variable is one chunk per
granule. Slots without a granule have no chunk references and read as fill values;
use `granule_present` to tell a missing granule from masked data.

Build model: write an empty template covering the full cube, then region-write one
cycle at a time. Region writes skip empty references, so a hole never erases an
existing reference. Granule coordinates in a written region are authoritative for
that region.

Science phase only (21-day orbit, 584 passes). Cal/Val data does not fit this axis.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import earthaccess
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from obspec_utils.readers import BlockStoreReader, EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import copy_and_replace_metadata
from virtualizarr.parsers import HDFParser

SHORT_NAME = "SWOT_L2_LR_SSH_EXPERT_D"
N_PASSES = 584
HTTPS_HOST = "https://archive.swot.podaac.earthdata.nasa.gov/"
HTTPS_PREFIX = HTTPS_HOST + "podaac-swot-ops-cumulus-protected/"
DROP_VARS = ["polarization_karin"]  # fixed-length char array; not virtualizable
GRID_VARS = ["latitude", "longitude"]  # stored once per pass (no cycle dim)
LEAD = ("cycle", "pass_number")

NAME_RE = re.compile(
    r"SWOT_L2_LR_SSH_Expert_(\d{3})_(\d{3})_\d{8}T\d{6}_\d{8}T\d{6}_(\w{4})_(\d{2})\.nc$"
)

# global attrs that are constant across granules; everything per-granule becomes a coord
KEEP_ATTRS = [
    "Conventions", "title", "institution", "source", "platform", "reference_document",
    "contact", "short_name", "product_version", "references", "wavelength",
    "ellipsoid_semi_major_axis", "ellipsoid_flattening",
]

# per-granule global attrs -> (cycle, pass_number) coordinates: name -> (attr, kind)
GRANULE_ATTRS = {
    "time_coverage_start": ("time_coverage_start", "time"),
    "time_coverage_end": ("time_coverage_end", "time"),
    "equator_time": ("equator_time", "time"),
    "equator_longitude": ("equator_longitude", "float"),
    "geospatial_lat_min": ("geospatial_lat_min", "float"),
    "geospatial_lat_max": ("geospatial_lat_max", "float"),
    "geospatial_lon_min": ("geospatial_lon_min", "float"),
    "geospatial_lon_max": ("geospatial_lon_max", "float"),
    "good_ocean_data_percent": ("good_ocean_data_percent", "float"),
}


# --------------------------------------------------------------------------- discovery

@dataclass(frozen=True)
class Granule:
    cycle: int
    pass_number: int
    crid: str
    counter: int
    url: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def find_granules(cycles: list[int], passes: list[int]) -> list[Granule]:
    """One granule per (cycle, pass): prefer PGD (reprocessed) over PID, then latest counter."""
    found: dict[tuple[int, int], Granule] = {}
    pass_set = set(passes)
    for c in cycles:
        for g in earthaccess.search_data(
            short_name=SHORT_NAME, granule_name=f"SWOT_L2_LR_SSH_Expert_{c:03d}_*.nc"
        ):
            url = g.data_links(access="external")[0]
            m = NAME_RE.search(url)
            if not m or int(m[2]) not in pass_set:
                continue
            gr = Granule(int(m[1]), int(m[2]), m[3], int(m[4]), url)
            key = (gr.cycle, gr.pass_number)
            if key not in found or _rank(gr) > _rank(found[key]):
                found[key] = gr
    return sorted(found.values(), key=lambda g: (g.cycle, g.pass_number))


def _rank(g: Granule) -> tuple[int, int]:
    return (g.crid.startswith("PGD"), g.counter)


# --------------------------------------------------------------------------- parsing

def make_registry(token: str) -> ObjectStoreRegistry:
    store = HTTPStore.from_url(
        HTTPS_HOST, client_options={"default_headers": {"Authorization": f"Bearer {token}"}}
    )
    return ObjectStoreRegistry({HTTPS_HOST: store})


def parse_granules(
    granules: list[Granule], token: str, workers: int = 8, eager: bool = True
) -> list[xr.Dataset]:
    """Parse granules into virtual datasets.

    eager=True fetches each whole file (~35 MB) in parallel ranges before parsing:
    ~12 s/granule over HTTPS outside AWS vs ~80 s for the default block reader,
    whose many small metadata reads each pay the Earthdata redirect. In-region
    (S3, low latency) the block reader avoids downloading whole files.
    Threads help only with eager: h5py's global lock serializes HDF5 parsing.
    """
    registry = make_registry(token)
    reader = EagerStoreReader if eager else BlockStoreReader
    parser = HDFParser(drop_variables=DROP_VARS, reader_factory=reader)

    def _open(g: Granule) -> xr.Dataset:
        return open_virtual_dataset(g.url, registry=registry, parser=parser, loadable_variables=[])

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_open, granules))


def check_consistent(vdss: list[xr.Dataset], granules: list[Granule]) -> None:
    """Every granule must match the first in variables, dims, zarr metadata and attrs."""
    ref = vdss[0]
    problems = []
    for vds, g in zip(vdss[1:], granules[1:]):
        if set(vds.variables) != set(ref.variables):
            problems.append(f"{g.filename}: variable set differs "
                            f"{set(vds.variables) ^ set(ref.variables)}")
            continue
        for name, rv in ref.variables.items():
            v = vds[name].variable
            if v.dims != rv.dims:
                problems.append(f"{g.filename}:{name}: dims {v.dims} != {rv.dims}")
            elif v.data.metadata.to_dict() != rv.data.metadata.to_dict():
                problems.append(f"{g.filename}:{name}: zarr metadata differs")
            elif v.attrs != rv.attrs or v.encoding != rv.encoding:
                problems.append(f"{g.filename}:{name}: attrs/encoding differ")
    if problems:
        raise ValueError("inconsistent granules:\n  " + "\n  ".join(problems[:50]))


# --------------------------------------------------------------------------- cube blocks

def _stack_manifest(
    ref: ManifestArray, members: np.ndarray, lead_dims: tuple[str, ...], dims: tuple[str, ...]
) -> ManifestArray:
    """Stack single-granule ManifestArrays into an array with extra leading dims.

    `members` is an object array of shape `lead_shape` holding ManifestArray or None.
    None slots become empty (null) chunk references.
    """
    lead_shape = members.shape
    grid = ref.manifest.shape_chunk_grid
    paths = np.full(lead_shape + grid, "", dtype=np.dtypes.StringDType())
    offsets = np.zeros(lead_shape + grid, dtype=np.uint64)
    lengths = np.zeros(lead_shape + grid, dtype=np.uint64)
    for idx in np.ndindex(lead_shape):
        ma = members[idx]
        if ma is not None:
            paths[idx] = ma.manifest._paths
            offsets[idx] = ma.manifest._offsets
            lengths[idx] = ma.manifest._lengths
    manifest = ChunkManifest.from_arrays(paths=paths, offsets=offsets, lengths=lengths)
    metadata = copy_and_replace_metadata(
        ref.metadata,
        new_shape=list(lead_shape) + list(ref.shape),
        new_chunks=[1] * len(lead_shape) + list(ref.metadata.chunks),
        new_dimension_names=lead_dims + dims,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def build_block(
    cycles: list[int],
    passes: list[int],
    slots: dict[tuple[int, int], tuple[Granule, xr.Dataset]],
    ref: xr.Dataset,
) -> xr.Dataset:
    """Virtual dataset for the block cycles x passes. Missing slots are null refs.

    `ref` is any parsed granule; it supplies metadata/attrs for every variable
    (check_consistent guarantees they agree).
    """
    nc, npass = len(cycles), len(passes)
    ci = {c: i for i, c in enumerate(cycles)}
    pi = {p: i for i, p in enumerate(passes)}

    variables = {}
    for name, rv in ref.variables.items():
        if name in GRID_VARS:
            members = np.full((npass,), None, dtype=object)
            for (c, p), (_, vds) in sorted(slots.items()):  # first cycle present wins
                if members[pi[p]] is None:
                    members[pi[p]] = vds[name].data
            lead = ("pass_number",)
        else:
            members = np.full((nc, npass), None, dtype=object)
            for (c, p), (_, vds) in slots.items():
                members[ci[c], pi[p]] = vds[name].data
            lead = LEAD
        variables[name] = xr.Variable(
            lead + rv.dims,
            _stack_manifest(rv.data, members, lead, rv.dims),
            attrs=rv.attrs,
            encoding=rv.encoding,
        )

    coords = {"cycle": ("cycle", np.asarray(cycles, dtype="int32")),
              "pass_number": ("pass_number", np.asarray(passes, dtype="int32"))}
    coords.update(_granule_coords(cycles, passes, slots))
    coord_names = set(ref.coords) | set(GRID_VARS)

    ds = xr.Dataset(
        {k: v for k, v in variables.items() if k not in coord_names},
        coords={**coords, **{k: v for k, v in variables.items() if k in coord_names}},
        attrs={k: ref.attrs[k] for k in KEEP_ATTRS if k in ref.attrs}
        | {"cube_layout": "(cycle, pass_number) science-phase cube; see swot_cube.py"},
    )
    return ds


def _granule_coords(cycles, passes, slots) -> dict:
    shape = (len(cycles), len(passes))
    ci = {c: i for i, c in enumerate(cycles)}
    pi = {p: i for i, p in enumerate(passes)}
    present = np.zeros(shape, bool)
    filename = np.full(shape, "", dtype=object)
    crid = np.full(shape, "", dtype=object)
    extra = {
        k: np.full(shape, np.datetime64("NaT", "ns") if kind == "time" else np.nan)
        for k, (_, kind) in GRANULE_ATTRS.items()
    }
    for (c, p), (g, vds) in slots.items():
        idx = (ci[c], pi[p])
        present[idx] = True
        filename[idx] = g.filename
        crid[idx] = g.crid
        for k, (attr, kind) in GRANULE_ATTRS.items():
            val = vds.attrs.get(attr)
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            if kind == "time":
                extra[k][idx] = pd.Timestamp(str(val).rstrip("Z")).to_datetime64()
            else:
                extra[k][idx] = float(val)
    out = {"granule_present": (LEAD, present), "filename": (LEAD, filename),
           "crid": (LEAD, crid)}
    # explicit encoding: float seconds with NaN fill, so NaT (holes) round-trips and
    # every region write encodes times identically
    time_enc = {"units": "seconds since 2000-01-01 00:00:00", "calendar": "proleptic_gregorian",
                "dtype": "float64", "_FillValue": np.nan}
    for k, v in extra.items():
        kind = GRANULE_ATTRS[k][1]
        out[k] = xr.Variable(LEAD, v, encoding=time_enc if kind == "time" else {})
    return out


# --------------------------------------------------------------------------- icechunk

def _container(headers: dict | None = None) -> icechunk.VirtualChunkContainer:
    return icechunk.VirtualChunkContainer(
        url_prefix=HTTPS_PREFIX, store=icechunk.http_store(headers=headers)
    )


def create_repo(path: str | Path, overwrite: bool = False) -> icechunk.Repository:
    """Repo config stores only the container prefix -- never credentials."""
    path = Path(path)
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(_container())
    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)), config)


def open_repo(path: str | Path, token: str) -> icechunk.Repository:
    """Open for reading; the Earthdata token is injected at runtime only."""
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        _container(headers={"authorization": f"Bearer {token}"})
    )
    return icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(path)),
        config=config,
        authorize_virtual_chunk_access={HTTPS_PREFIX: icechunk.Credentials.HttpAccess()},
    )


def open_cube(path: str | Path, token: str, branch: str = "main") -> xr.Dataset:
    repo = open_repo(path, token)
    return xr.open_zarr(repo.readonly_session(branch).store, consolidated=False, zarr_format=3)


# --------------------------------------------------------------------------- build

def build(repo_path: str, cycles: list[int], passes: list[int],
          drop: set[tuple[int, int]] = frozenset(), overwrite: bool = False) -> None:
    auth = earthaccess.login(strategy="netrc")
    token = auth.token["access_token"]

    granules = [g for g in find_granules(cycles, passes)
                if (g.cycle, g.pass_number) not in drop]
    print(f"found {len(granules)} granules for {len(cycles)} cycles x {len(passes)} passes"
          f" (dropped {len(drop)} on purpose)")
    vdss = parse_granules(granules, token)
    check_consistent(vdss, granules)
    ref = vdss[0]
    slots = {(g.cycle, g.pass_number): (g, v) for g, v in zip(granules, vdss)}

    repo = create_repo(repo_path, overwrite=overwrite)
    all_passes = list(range(1, N_PASSES + 1))

    # 1. template: full extent, no references at all
    session = repo.writable_session("main")
    build_block(cycles, all_passes, {}, ref).vz.to_icechunk(session.store, mode="w")
    session.commit("template: empty (cycle, pass_number) cube")

    # 2. one region write per cycle
    session = repo.writable_session("main")
    for i, c in enumerate(cycles):
        cyc_slots = {k: v for k, v in slots.items() if k[0] == c}
        block = build_block([c], all_passes, cyc_slots, ref)
        block.vz.to_icechunk(
            session.store,
            region={"cycle": slice(i, i + 1), "pass_number": slice(0, N_PASSES)},
        )
        print(f"  cycle {c}: wrote {len(cyc_slots)} granules")
    print("commit:", session.commit(f"fill cycles {cycles[0]}-{cycles[-1]}"))


# --------------------------------------------------------------------------- update

def _runs(values: list[int]) -> list[list[int]]:
    """Split sorted ints into contiguous runs: [1,2,3,7,8] -> [[1,2,3],[7,8]]."""
    runs: list[list[int]] = []
    for v in sorted(values):
        if runs and v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


def _rank_filename(fname: str) -> tuple[int, int] | None:
    m = NAME_RE.search(fname)
    return (m[3].startswith("PGD"), int(m[4])) if m else None


def update(repo_path: str, granules: list[Granule], message: str | None = None) -> str | None:
    """Add newly published granules to an existing cube in one commit.

    - Cycles beyond the cube's last cycle are appended as empty slabs (append_dim="cycle"),
      including any skipped cycles so the cycle axis stays contiguous.
    - Each granule is region-written into its slot; runs of consecutive passes in a cycle
      share one region write (a day is ~28 consecutive passes, two runs if it spans a
      cycle boundary).
    - A slot that already holds a granule is replaced only if the new one ranks higher
      (PGD over PID, then higher product counter); otherwise the granule is skipped.
    """
    token = earthaccess.login(strategy="netrc").token["access_token"]
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(repo_path)))
    session = repo.writable_session("main")
    # read only the small native coords: opening with xarray would decode `time`,
    # which touches a virtual chunk and needs Earthdata auth
    root = zarr.open_group(session.store, mode="r")
    cycles = [int(c) for c in root["cycle"][:]]
    current = {
        (int(c), int(p)): str(f)
        for c, row in zip(cycles, root["filename"][:])
        for p, f in zip(root["pass_number"][:], row)
        if f
    }

    todo = []
    for g in granules:
        old = current.get((g.cycle, g.pass_number))
        if old is None or _rank(g) > _rank_filename(old):
            todo.append(g)
    if not todo:
        print("nothing new")
        return None
    if min(g.cycle for g in todo) < cycles[0]:
        raise ValueError("granules precede the cube's first cycle; prepending is not supported")

    vdss = parse_granules(todo, token)
    check_consistent(vdss, todo)
    # compatibility with the arrays already in the cube is enforced by VirtualiZarr's
    # append/region writers (check_compatible_arrays / check_compatible_encodings)
    ref = vdss[0]
    slots = {(g.cycle, g.pass_number): (g, v) for g, v in zip(todo, vdss)}
    all_passes = list(range(1, N_PASSES + 1))

    # 1. grow the cycle axis with empty slabs
    new_cycles = list(range(cycles[-1] + 1, max(g.cycle for g in todo) + 1))
    for c in new_cycles:
        build_block([c], all_passes, {}, ref).vz.to_icechunk(session.store, append_dim="cycle")
    cycles += new_cycles
    ci = {c: i for i, c in enumerate(cycles)}

    # 2. region-write each run of consecutive passes
    for c in sorted({k[0] for k in slots}):
        for run in _runs([p for (cc, p) in slots if cc == c]):
            run_slots = {(c, p): slots[(c, p)] for p in run}
            build_block([c], run, run_slots, ref).vz.to_icechunk(
                session.store,
                region={"cycle": slice(ci[c], ci[c] + 1),
                        "pass_number": slice(run[0] - 1, run[-1])},
            )
            print(f"  cycle {c} passes {run[0]}-{run[-1]}: wrote {len(run)} granules")

    msg = message or (f"update: {len(todo)} granules"
                      + (f", new cycles {new_cycles}" if new_cycles else ""))
    return session.commit(msg)


# --------------------------------------------------------------------------- CLI

def _parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="create a cube from scratch")
    b.add_argument("--cycles", default="10-13", help="e.g. 10-13 or 10,12")
    b.add_argument("--passes", default="1-16", help="passes to populate (cube always has 584)")
    b.add_argument("--drop", default="", help="cycle:pass slots to leave empty, e.g. 11:5,12:9")
    b.add_argument("--overwrite", action="store_true")
    u = sub.add_parser("update", help="add granules to an existing cube")
    u.add_argument("--cycles", required=True)
    u.add_argument("--passes", default=f"1-{N_PASSES}")
    for p in (b, u):
        p.add_argument("--repo", default="data/swot_expert_d_cube.icechunk")
    a = ap.parse_args()
    if a.cmd == "build":
        drop = {tuple(map(int, s.split(":"))) for s in a.drop.split(",") if s}
        build(a.repo, _parse_range(a.cycles), _parse_range(a.passes), drop, a.overwrite)
    else:
        found = find_granules(_parse_range(a.cycles), _parse_range(a.passes))
        print("commit:", update(a.repo, found))
