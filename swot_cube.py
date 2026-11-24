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

Build model: write an empty template covering the full cube, then region-write
runs of consecutive passes, located by coordinate labels (region="auto"). Blocks are
assembled with expand_dims / combine_by_coords; the template uses empty manifests. Region writes skip empty references, so a hole never erases an
existing reference. Granule coordinates in a written region are authoritative for
that region.

References are relative: every chunk location is stored as
``vcc://podaac-swot/<key>`` and resolved through the virtual chunk container
named ``podaac-swot``. The repo stores an S3 container under that name (no
credentials); readers outside AWS override the same-named container with the
HTTPS archive at open time. One store therefore serves S3 or HTTPS access.

Science phase only (21-day orbit, 584 passes). Cal/Val data does not fit this axis.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import earthaccess  # CMR granule search only; auth comes from earthaccess-auth
import earthaccess_auth
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from earthaccess_auth import Auth
from earthaccess_auth.adapters.icechunk import earthdata_containers_credentials
from earthaccess_auth.adapters.obstore import http_client_options
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
S3_BUCKET = "podaac-swot-ops-cumulus-protected"
HTTPS_PREFIX = HTTPS_HOST + S3_BUCKET + "/"
S3_PREFIX = f"s3://{S3_BUCKET}/"
VCC_NAME = "podaac-swot"            # container name; refs are relative to it
VCC_ROOT = f"vcc://{VCC_NAME}/"
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

def login(strategy: str = "netrc") -> Auth:
    """Earthdata Login via earthaccess-auth (strategies: netrc, environment, interactive, all)."""
    return earthaccess_auth.login(strategy=strategy)


def make_registry(auth: Auth) -> ObjectStoreRegistry:
    """obstore HTTPS store carrying the earthaccess-auth bearer header."""
    store = HTTPStore.from_url(HTTPS_HOST, client_options=http_client_options(auth))
    return ObjectStoreRegistry({HTTPS_HOST: store})


def parse_granules(
    granules: list[Granule], auth: Auth, workers: int = 8, eager: bool = True
) -> list[xr.Dataset]:
    """Parse granules into virtual datasets.

    eager=True fetches each whole file (~35 MB) in parallel ranges before parsing:
    ~12 s/granule over HTTPS outside AWS vs ~80 s for the default block reader,
    whose many small metadata reads each pay the Earthdata redirect. In-region
    (S3, low latency) the block reader avoids downloading whole files.
    Threads help only with eager: h5py's global lock serializes HDF5 parsing.
    """
    registry = make_registry(auth)
    reader = EagerStoreReader if eager else BlockStoreReader
    parser = HDFParser(drop_variables=DROP_VARS, reader_factory=reader)

    def _open(g: Granule) -> xr.Dataset:
        return open_virtual_dataset(g.url, registry=registry, parser=parser, loadable_variables=[])

    with ThreadPoolExecutor(workers) as ex:
        return [to_relative(v) for v in ex.map(_open, granules)]


def to_relative(vds: xr.Dataset) -> xr.Dataset:
    """Rewrite absolute S3/HTTPS chunk paths to vcc://podaac-swot/<key>."""
    def _rel(path: str) -> str:
        for prefix in (HTTPS_PREFIX, S3_PREFIX):
            if path.startswith(prefix):
                return VCC_ROOT + path[len(prefix):]
        raise ValueError(f"chunk path outside the SWOT bucket: {path}")
    return vds.vz.rename_paths(_rel)


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
#
# Built from public xarray / VirtualiZarr operations:
#   lift        -- expand_dims gives one granule its (cycle, pass_number) slot
#   combine     -- xr.combine_by_coords stacks lifted granules of consecutive passes
#   empty_cube  -- the full-extent template: every slot a null chunk reference
# Holes are never padded in: the template already holds null references everywhere,
# and region writes (region="auto", located by label) only touch the slots they cover.
# (Padding with reindex also works -- VirtualiZarr turns alignment fills into null
# references -- but xarray and VirtualiZarr build element-level masks for it, which is
# prohibitive at cube scale: ~4e9 elements per variable for 10 cycles x 584 passes.)
# While blocks are assembled, coordinates are held as data variables; write()
# restores them as coordinates.

CUBE_COORDS = ["latitude", "longitude", "latitude_nadir", "longitude_nadir",
               "granule_present", "filename", "crid", *GRANULE_ATTRS]

# fill values for empty slots (virtual arrays always read their own fill value)
GRANULE_FILL = {"granule_present": False, "filename": "", "crid": ""} | {
    k: (np.datetime64("NaT", "ns") if kind == "time" else np.nan)
    for k, (_, kind) in GRANULE_ATTRS.items()
}

# explicit encoding: float seconds with NaN fill, so NaT (holes) round-trips and
# every region write encodes times identically
TIME_ENC = {"units": "seconds since 2000-01-01 00:00:00", "calendar": "proleptic_gregorian",
            "dtype": "float64", "_FillValue": np.nan}


def _granule_fields(g: Granule, vds: xr.Dataset) -> dict:
    """Per-granule values (from the filename and global attrs) as (1, 1) arrays."""
    vals = {"granule_present": True, "filename": g.filename, "crid": g.crid}
    for k, (attr, kind) in GRANULE_ATTRS.items():
        val = vds.attrs.get(attr)
        if val is None or (isinstance(val, str) and not val.strip()):
            vals[k] = GRANULE_FILL[k]
        elif kind == "time":
            vals[k] = np.datetime64(pd.Timestamp(str(val).rstrip("Z")).to_datetime64(), "ns")
        else:
            vals[k] = float(val)
    return {k: (LEAD, np.array([[v]], dtype=object if isinstance(v, str) else None))
            for k, v in vals.items()}


def lift(g: Granule, vds: xr.Dataset) -> xr.Dataset:
    """One parsed granule -> a (1, 1) block at its (cycle, pass_number) slot.

    Swath lat/lon get only the pass_number dim (fixed grid per pass).
    Dataset.expand_dims skips coordinates, so they are demoted to data vars first.
    """
    flat = vds.reset_coords()
    grid = flat[GRID_VARS].expand_dims(pass_number=[g.pass_number])
    rest = flat.drop_vars(GRID_VARS).expand_dims(cycle=[g.cycle], pass_number=[g.pass_number])
    out = xr.merge([rest, grid], join="exact", combine_attrs="drop_conflicts")
    out = out.assign(_granule_fields(g, vds))
    out.attrs = {k: vds.attrs[k] for k in KEEP_ATTRS if k in vds.attrs} | {
        "cube_layout": "(cycle, pass_number) science-phase cube; see swot_cube.py"}
    return out


def combine(blocks: list[xr.Dataset]) -> xr.Dataset:
    """Stack lifted granules by their cycle/pass_number labels (outer join: gaps are holes)."""
    return xr.combine_by_coords(
        blocks, data_vars="all", coords="minimal", compat="override",
        join="outer", combine_attrs="override", fill_value=GRANULE_FILL,
    )


def empty_cube(like: xr.Dataset, cycles: list[int], passes: list[int]) -> xr.Dataset:
    """Template covering cycles x passes with no chunk references at all.

    `like` is any lifted granule. Virtual arrays keep its zarr metadata (codecs, chunks)
    with the leading dims resized and an empty manifest; granule fields get fill values.
    Cost scales with the number of chunks, not elements.
    """
    sizes = {"cycle": len(cycles), "pass_number": len(passes)}
    variables = {}
    for name, var in like.data_vars.items():
        shape = tuple(sizes.get(d, n) for d, n in zip(var.dims, var.shape))
        if isinstance(var.data, ManifestArray):
            meta = copy_and_replace_metadata(var.data.metadata, new_shape=list(shape))
            # the granule's own chunk grid, with the cycle/pass_number entries resized
            grid = tuple(sizes.get(d, g) for d, g in zip(var.dims, var.data.manifest.shape_chunk_grid))
            data = ManifestArray(metadata=meta, chunkmanifest=ChunkManifest(entries={}, shape=grid))
        else:
            data = np.full(shape, GRANULE_FILL[name], dtype=var.dtype)
        variables[name] = xr.Variable(var.dims, data, attrs=var.attrs, encoding=var.encoding)
    return xr.Dataset(variables, coords={"cycle": cycles, "pass_number": passes}, attrs=like.attrs)


def _finalize(block: xr.Dataset) -> xr.Dataset:
    """Restore coordinates and pin encodings before writing."""
    out = block.set_coords([c for c in CUBE_COORDS if c in block])
    for k, (_, kind) in GRANULE_ATTRS.items():
        if kind == "time" and k in out:
            out[k].encoding = dict(TIME_ENC)
    return out


# --------------------------------------------------------------------------- icechunk

def _s3_container() -> icechunk.VirtualChunkContainer:
    """Canonical container saved in the repo: S3, no credentials."""
    return icechunk.VirtualChunkContainer(
        url_prefix=S3_PREFIX, store=icechunk.s3_store(region="us-west-2"), name=VCC_NAME
    )


def _https_container(auth: Auth) -> icechunk.VirtualChunkContainer:
    """Read-time override of the same-named container: HTTPS archive + EDL bearer header.

    Never persist a config holding this container, and never print/display the
    repo config after applying it -- its repr shows the bearer token.
    """
    return icechunk.VirtualChunkContainer(
        url_prefix=HTTPS_PREFIX,
        store=icechunk.http_store(headers=http_client_options(auth)["default_headers"]),
        name=VCC_NAME,
    )


def _assert_relative(vds: xr.Dataset) -> None:
    for name, var in vds.variables.items():
        if isinstance(var.data, ManifestArray):
            for path in var.data.manifest.iter_nonempty_paths():
                if not path.startswith(VCC_ROOT):
                    raise ValueError(f"{name}: absolute chunk path {path!r}; expected {VCC_ROOT}...")


def write(vds: xr.Dataset, store: icechunk.IcechunkStore, **kwargs) -> None:
    """to_icechunk for relative refs.

    VirtualiZarr's container validation does not understand vcc:// locations, so it
    is disabled and replaced by a check that every reference is relative to the
    podaac-swot container.
    """
    vds = _finalize(vds)
    _assert_relative(vds)
    vds.vz.to_icechunk(store, validate_containers=False, **kwargs)


def create_repo(path: str | Path, overwrite: bool = False) -> icechunk.Repository:
    """Repo config stores only the container prefix -- never credentials."""
    path = Path(path)
    if overwrite and path.exists():
        import shutil
        shutil.rmtree(path)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(_s3_container())
    return icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)), config)


def open_repo(path: str | Path, auth: Auth, access: str = "https") -> icechunk.Repository:
    """Open for reading over "https" (anywhere) or "s3" (in us-west-2 only).

    https: the podaac-swot container is overridden with the HTTPS archive and the
        earthaccess-auth bearer header, via a config override that is never
        persisted. Headers are static: EDL tokens last ~60 days, so long-lived
        readers should log in again and reopen.
    s3: the stored S3 container is used as-is, authorized with earthaccess-auth's
        refreshable per-bucket credentials. (Untested outside us-west-2.)
    """
    storage = icechunk.local_filesystem_storage(str(path))
    if access == "https":
        config = icechunk.RepositoryConfig.default()
        config.set_virtual_chunk_container(_https_container(auth))
        return icechunk.Repository.open(
            storage, config=config,
            authorize_virtual_chunk_access={HTTPS_PREFIX: icechunk.Credentials.HttpAccess()},
        )
    if access == "s3":
        earthaccess_auth.set_default_auth(auth)  # identity behind the S3 credential fetches
        repo = icechunk.Repository.open(storage)
        return repo.reopen(authorize_virtual_chunk_access=earthdata_containers_credentials(repo))
    raise ValueError(f"access must be 'https' or 's3', not {access!r}")


def open_cube(path: str | Path, auth: Auth, branch: str = "main", access: str = "https") -> xr.Dataset:
    repo = open_repo(path, auth, access)
    return xr.open_zarr(repo.readonly_session(branch).store, consolidated=False, zarr_format=3)


def write_runs(store: icechunk.IcechunkStore, lifted: dict[tuple[int, int], xr.Dataset]) -> None:
    """Region-write lifted granules, one write per run of consecutive passes in a cycle.

    Slots are located by label (region="auto"). Runs are never padded, so slots between
    runs -- holes, or granules already in the cube -- and their granule coords are not touched.
    """
    for c in sorted({k[0] for k in lifted}):
        for run in _runs([p for (cc, p) in lifted if cc == c]):
            write(combine([lifted[(c, p)] for p in run]), store,
                  region={"cycle": "auto", "pass_number": "auto"})
            print(f"  cycle {c} passes {run[0]}-{run[-1]}: wrote {len(run)} granules")


# --------------------------------------------------------------------------- build

def build(repo_path: str, cycles: list[int], passes: list[int],
          drop: set[tuple[int, int]] = frozenset(), overwrite: bool = False) -> None:
    auth = login()

    granules = [g for g in find_granules(cycles, passes)
                if (g.cycle, g.pass_number) not in drop]
    print(f"found {len(granules)} granules for {len(cycles)} cycles x {len(passes)} passes"
          f" (dropped {len(drop)} on purpose)")
    vdss = parse_granules(granules, auth)
    check_consistent(vdss, granules)
    lifted = [lift(g, v) for g, v in zip(granules, vdss)]

    repo = create_repo(repo_path, overwrite=overwrite)
    all_passes = list(range(1, N_PASSES + 1))

    # 1. template: every cycle x all 584 passes, no references
    session = repo.writable_session("main")
    write(empty_cube(lifted[0], cycles, all_passes), session.store, mode="w")
    session.commit("template: empty (cycle, pass_number) cube")

    # 2. per cycle, per run of consecutive passes: combine and write by label
    session = repo.writable_session("main")
    by_slot = {(g.cycle, g.pass_number): b for g, b in zip(granules, lifted)}
    write_runs(session.store, by_slot)
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
    auth = login()
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

    vdss = parse_granules(todo, auth)
    check_consistent(vdss, todo)
    # compatibility with the arrays already in the cube is enforced by VirtualiZarr's
    # append/region writers (check_compatible_arrays / check_compatible_encodings)
    lifted = {(g.cycle, g.pass_number): lift(g, v) for g, v in zip(todo, vdss)}
    all_passes = list(range(1, N_PASSES + 1))

    # 1. grow the cycle axis with empty slabs
    new_cycles = list(range(cycles[-1] + 1, max(g.cycle for g in todo) + 1))
    like = next(iter(lifted.values()))
    for c in new_cycles:
        write(empty_cube(like, [c], all_passes), session.store, append_dim="cycle")

    # 2. region-write each run of consecutive passes, located by label
    write_runs(session.store, lifted)

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
