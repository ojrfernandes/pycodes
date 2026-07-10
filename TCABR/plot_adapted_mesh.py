#!/home/jfernandes/.venv/bin/python
"""
Visualize an M3D-C1 adapted mesh straight from the PUMI VTK output
(m3dc1_mesh_write option 0/3, i.e. iadapt_writevtk=1), with no PUMI,
VTK, meshio, or pyvista dependency -- only numpy + matplotlib.

PUMI's apf::writeVtkFiles writes a directory:
    <prefix>/<prefix>.pvtu          <- manifest, lists one <Piece Source=".../N.vtu"/> per rank
    <prefix>/<rank>/<rank>.vtu      <- one serial VTK XML UnstructuredGrid per rank

Each DataArray with format="binary" stores its content as TWO
concatenated base64 blocks with no separator: first the base64 of a
single header-typed integer (byte count of the payload -- 8 base64
characters for the default header_type="UInt32"), then the base64 of
the raw payload bytes. This is decoded manually in _decode_array.

Cell data caveat: the runtime m3dc1_mesh_write (option 0, used by
iadapt_writevtk=1) only ever writes "apf_part" (MPI partition id) as
cell data -- there is no per-element geometric face/zone id available
this way, so --color-by face/elem will only work on VTK files written
by m3dc1_mfmgen (which explicitly adds "gface_1"/"elem_1" tags before
writing), not on adapt-run output. Getting a face id at runtime would
require m3dc1_mesh_write option 3 (adapt.f90:267, change the "0" to a
"3"), which calls output_face_data() to write a *separate* legacy-ASCII
VTK file named "geoId_<rank>.vtk" -- note this call is compiled out
unless the m3dc1_scorec build defines USEVTK, and its format isn't
supported by this script (a different parser would be needed to read
and merge it by cell order with the main dump).
"""
import argparse
import base64
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_VTK_NP = {
    'Int8': 'i1', 'UInt8': 'u1',
    'Int16': 'i2', 'UInt16': 'u2',
    'Int32': 'i4', 'UInt32': 'u4',
    'Int64': 'i8', 'UInt64': 'u8',
    'Float32': 'f4', 'Float64': 'f8',
}

_COLOR_FIELDS = {'face': 'gface_1', 'part': 'apf_part', 'elem': 'elem_1'}


def _decode_array(text: str, dtype: np.dtype, header_dtype: np.dtype) -> np.ndarray:
    """Decode one binary-format VTK XML DataArray's text content."""
    text = ''.join(text.split())
    hlen = math.ceil(header_dtype.itemsize / 3) * 4
    nbytes = np.frombuffer(base64.b64decode(text[:hlen]), dtype=header_dtype)[0]
    payload = base64.b64decode(text[hlen:])
    if len(payload) != nbytes:
        raise ValueError(f'decoded payload length {len(payload)} != header byte count {nbytes}')
    return np.frombuffer(payload, dtype=dtype)


def _read_vtu_piece(vtu_path: Path) -> dict:
    root = ET.parse(vtu_path).getroot()
    byte_order = '<' if root.attrib.get('byte_order', 'LittleEndian') == 'LittleEndian' else '>'
    header_dtype = np.dtype(byte_order + _VTK_NP[root.attrib.get('header_type', 'UInt32')])

    def get_array(parent_tag: str, name: str):
        da = root.find(f'.//{parent_tag}/DataArray[@Name="{name}"]')
        if da is None or not da.text:
            return None
        dtype = np.dtype(byte_order + _VTK_NP[da.attrib['type']])
        arr = _decode_array(da.text, dtype, header_dtype)
        ncomp = int(da.attrib.get('NumberOfComponents', 1))
        return arr.reshape(-1, ncomp) if ncomp > 1 else arr

    points = get_array('Points', 'coordinates')
    connectivity = get_array('Cells', 'connectivity')
    offsets = get_array('Cells', 'offsets')
    types = get_array('Cells', 'types')

    cell_data = {}
    cell_data_elem = root.find('.//CellData')
    if cell_data_elem is not None:
        for da in cell_data_elem.findall('DataArray'):
            cell_data[da.attrib['Name']] = get_array('CellData', da.attrib['Name'])

    return dict(points=points, connectivity=connectivity, offsets=offsets,
                types=types, cell_data=cell_data)


def _pieces_from_pvtu(pvtu_path: Path) -> list[Path]:
    root = ET.parse(pvtu_path).getroot()
    base = pvtu_path.parent
    return [(base / piece.attrib['Source']).resolve() for piece in root.findall('.//Piece')]


def _find_pieces(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == '.pvtu':
        return _pieces_from_pvtu(p)
    if p.is_file() and p.suffix == '.vtu':
        return [p]
    if p.is_dir():
        pvtus = sorted(p.glob('*.pvtu'))
        if pvtus:
            return _pieces_from_pvtu(pvtus[0])
        vtus = sorted(p.glob('**/*.vtu'))
        if vtus:
            return vtus
    for cand in (p.with_suffix('.pvtu'), p / f'{p.name}.pvtu'):
        if cand.is_file():
            return _pieces_from_pvtu(cand)
    raise FileNotFoundError(
        f'No .pvtu manifest or .vtu piece found for "{path}". Pass the VTK output '
        'directory/prefix exactly as given to m3dc1_mesh_write (iadapt_writevtk=1).'
    )


def _load_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read all pieces and return concatenated (R, Z, triangles, cell_data)."""
    pieces = _find_pieces(path)
    R_all, Z_all, tri_all = [], [], []
    cell_data_all: dict[str, list[np.ndarray]] = {}
    point_offset = 0

    for piece_path in pieces:
        d = _read_vtu_piece(piece_path)
        pts, conn, offs, types = d['points'], d['connectivity'], d['offsets'], d['types']

        is_tri = (types == 5)
        starts = np.concatenate(([0], offs[:-1]))
        counts = offs - starts
        uniform_tri = is_tri.all() and (counts == 3).all()

        if uniform_tri:
            tris = conn.reshape(-1, 3)
            valid = np.ones(len(tris), dtype=bool)
        else:
            warnings.warn(f'{piece_path}: non-triangle cells present; skipping those elements')
            valid = is_tri & (counts == 3)
            tris = np.array([conn[s:s + 3] for s, keep in zip(starts, valid) if keep], dtype=np.int64)
            tris = tris.reshape(-1, 3) if tris.size else np.zeros((0, 3), dtype=np.int64)

        tri_all.append(tris + point_offset)
        R_all.append(pts[:, 0])
        Z_all.append(pts[:, 1])
        for name, arr in d['cell_data'].items():
            if arr is not None:
                cell_data_all.setdefault(name, []).append(arr[valid])
        point_offset += pts.shape[0]

    R = np.concatenate(R_all)
    Z = np.concatenate(Z_all)
    triangles = np.concatenate(tri_all, axis=0)
    cell_data = {name: np.concatenate(chunks) for name, chunks in cell_data_all.items()}
    return R, Z, triangles, cell_data


def plot_adapted_mesh(path: str,
                       color_by: str | None = None,
                       xlim: tuple[float, float] | None = None,
                       zlim: tuple[float, float] | None = None,
                       linewidth: float = 0.3,
                       meshcolor: str = 'k',
                       cmap: str = 'tab20',
                       show_edges: bool = True,
                       ax=None,
                       dpi=100,
                       save: str | None = None) -> None:
    """
    Plot an M3D-C1 adapted mesh read directly from PUMI VTK output.

    Reads the mesh geometry (R, Z, triangle connectivity) and per-element
    fields written by apf::writeVtkFiles, with no dependency on PUMI, VTK,
    meshio, or pyvista, and no M3D-C1 run/equilibrium needed -- this can be
    called immediately after adapt_by_psi writes the VTK output
    (iadapt_writevtk=1), before initial_conditions is re-derived.

    Parameters
    ----------
    path : str
        Path to the VTK output as given to m3dc1_mesh_write: the output
        directory (e.g. "ts0000-adapted0"), a ".pvtu" manifest, or a single
        ".vtu" piece.
    color_by : {'face', 'part', 'elem', None}
        If 'face', color elements by geometric model face id (PUMI's
        "gface_1" cell field) -- useful to confirm exactly which zones an
        iadaptFaceNumber-restricted adapt touched. 'part' colors by MPI
        partition/rank ("apf_part"), 'elem' by PUMI's local element id
        ("elem_1"). None (default) draws a plain wireframe.
    xlim, zlim : (float, float), optional
        R and Z axis limits, e.g. to zoom on a vessel corner.
    linewidth : float
        Mesh/element edge line width.
    meshcolor : str
        Line color for the wireframe (or element edges when color_by is set).
    cmap : str
        Colormap used when color_by is set.
    show_edges : bool
        If color_by is set, also draw element edges on top of the fill.
    ax : matplotlib Axes, optional
        Existing axes to plot into. If None, a new figure is created.
    save : str, optional
        If given, save the figure to this path instead of showing it
        interactively (only when ax is None).
    """
    R, Z, triangles, cell_data = _load_mesh(path)
    triang = mtri.Triangulation(R, Z, triangles)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 8), dpi=dpi)
    else:
        fig = ax.figure

    if color_by is None:
        ax.triplot(triang, lw=linewidth, color=meshcolor)
    else:
        if color_by not in cell_data:
            raise KeyError(
                f'"{color_by}" not found in this VTK output; available cell data: '
                f'{list(cell_data)}. Note: "gface_1" (face id) and "elem_1" are only '
                f'present in mesh files written by m3dc1_mfmgen -- the runtime '
                f'm3dc1_mesh_write (iadapt_writevtk=1) only writes "apf_part". Use '
                f'--color-by part instead, or see the note in the module docstring '
                f'about getting per-face data from m3dc1_mesh_write option 3.'
            )
        data = cell_data[color_by].astype(float)
        tpc = ax.tripcolor(triang, facecolors=data, cmap=cmap,
                            edgecolors=meshcolor if show_edges else 'none',
                            linewidth=linewidth)
        fig.colorbar(tpc, ax=ax, label=color_by)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$Z$')
    ax.set_title(f'{Path(path).name}: {len(triangles)} elements, {len(R)} nodes')
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if zlim is not None:
        ax.set_ylim(*zlim)