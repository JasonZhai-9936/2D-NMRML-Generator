"""Predict a 1H-13C HSQC spectrum from assigned 1D 1H and 13C spectra.

A port of jsv2's predict_hsqc (JSpectraViewer): each assigned carbon with directly
bonded hydrogens gives one cross peak at (1H shift of those hydrogens, 13C shift of the
carbon), drawn as a 2D Gaussian with height proportional to the number of hydrogens.
Quaternary carbons give no cross peak.
"""
import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

HSQC_DEFAULTS = {
    'points_1h': 1024,   # direct dimension (F2) points
    'points_13c': 512,   # indirect dimension (F1) points
    'width_1h': 0.02,    # Gaussian standard deviation, ppm
    'width_13c': 0.5,    # Gaussian standard deviation, ppm
}


class PredictionError(ValueError):
    pass


def _to_fixed(value, places):
    """JavaScript +value.toFixed(places): round half away from zero on the exact binary value."""
    return float(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _cluster_center(cluster):
    peaks = cluster.get('peaks') or []
    return sum(peaks) / len(peaks) if peaks else float('nan')


def _same_structure(a, b):
    ids_a, ids_b = sorted(a['atoms']), sorted(b['atoms'])
    return ids_a == ids_b and all(a['atoms'][i]['element'] == b['atoms'][i]['element'] for i in ids_a)


def _axis(values, centers, margin, points, fallback):
    """High-to-low ppm axis over the 1D spectrum's range, widened to hold every cross
    peak (plus a margin)."""
    source = values if values is not None and len(values) else fallback
    lo, hi = math.inf, -math.inf
    for v in source:
        v = float(v)
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    for c in centers:
        lo = min(lo, c - margin)
        hi = max(hi, c + margin)
    step = (hi - lo) / (points - 1)
    return np.array([hi - i * step for i in range(points)])


def predict_hsqc(proton, carbon, options=None):
    """Predict an HSQC from two parsed 1D spectra (read_1d_nmrml results).

    Returns {'x', 'y', 'z' (shape len(y) x len(x)), 'id', 'name', 'experiment',
    'cross_peaks', 'assignment'}, ready for build_2d_nmrml.
    """
    opts = dict(HSQC_DEFAULTS, **(options or {}))
    h_data = proton and proton.get('assignment')
    c_data = carbon and carbon.get('assignment')
    if not h_data or not c_data:
        raise PredictionError('Predicted HSQC needs assigned 1D 1H and 13C spectra (nmrML with peak assignments).')
    structure = c_data.get('structure') or h_data.get('structure')
    if not structure:
        raise PredictionError('Predicted HSQC needs a molecular structure in the 1D nmrML files.')
    h_structure = h_data.get('structure')
    if h_structure and not _same_structure(structure, h_structure):
        raise PredictionError('The 1H and 13C files describe different structures (atom numbering differs); '
                              'they must be for the same compound with the same atom numbering.')
    atoms = structure['atoms']

    # Hydrogens bonded to each carbon
    bonded_hydrogens = {}
    for bond in structure['bonds']:
        for heavy, h in ((bond['a1'], bond['a2']), (bond['a2'], bond['a1'])):
            if heavy in atoms and h in atoms and atoms[heavy]['element'] == 'C' and atoms[h]['element'] == 'H':
                bonded_hydrogens.setdefault(heavy, []).append(h)

    h_clusters = [{'center': _cluster_center(c), 'atom_refs': c.get('atom_refs') or []}
                  for c in h_data.get('clusters') or []]
    h_clusters = [c for c in h_clusters if not math.isnan(c['center'])]

    cross_peaks = []
    for c in c_data.get('clusters') or []:
        c_center = _cluster_center(c)
        if math.isnan(c_center):
            continue
        for carbon_id in c.get('atom_refs') or []:
            if carbon_id not in atoms or atoms[carbon_id]['element'] != 'C':
                continue
            hydrogens = bonded_hydrogens.get(carbon_id, [])
            if not hydrogens:
                continue                      # quaternary carbon: no HSQC cross peak
            # 1H clusters assigned to these hydrogens (or to the carbon itself)
            for h in h_clusters:
                matched = [i for i in hydrogens if i in h['atom_refs']]
                if not matched and carbon_id not in h['atom_refs']:
                    continue
                h_atoms = matched or hydrogens
                cross_peaks.append({
                    'x': {'center': _to_fixed(h['center'], 4), 'atom_refs': ['H' + i for i in h_atoms]},
                    'y': {'center': _to_fixed(c_center, 3), 'atom_refs': ['C' + carbon_id]},
                    'intensity': len(h_atoms),
                })
    if not cross_peaks:
        raise PredictionError('No cross peaks found: no assigned carbon has bonded hydrogens with an assigned 1H signal.')

    x = _axis(proton.get('x'), [cp['x']['center'] for cp in cross_peaks], 0.5, opts['points_1h'], [-0.5, 12])
    y = _axis(carbon.get('x'), [cp['y']['center'] for cp in cross_peaks], 10, opts['points_13c'], [-5, 220])

    # Gaussian cross peaks; widths at least 1.5 grid points so every peak is resolved
    sx = max(opts['width_1h'], 1.5 * abs(x[1] - x[0]))
    sy = max(opts['width_13c'], 1.5 * abs(y[1] - y[0]))
    z = np.zeros((len(y), len(x)))
    for cp in cross_peaks:
        dy = (y - cp['y']['center']) / sy
        dx = (x - cp['x']['center']) / sx
        rows, cols = np.abs(dy) <= 4, np.abs(dx) <= 4      # only points within 4 standard deviations
        if rows.any() and cols.any():
            ddy, ddx = dy[rows][:, None], dx[cols][None, :]
            z[np.ix_(rows, cols)] += cp['intensity'] * np.exp(-0.5 * (ddx * ddx + ddy * ddy))

    return {
        'x': x, 'y': y, 'z': z,
        'id': '2D-1H-13C-HSQC-predicted',
        'name': c_data.get('name') or h_data.get('name') or 'Compound',
        'experiment': 'predicted 1H-13C HSQC',
        'cross_peaks': cross_peaks,
        'assignment': {
            'atoms': list(atoms.values()),
            'bonds': structure['bonds'],
            'cross_peaks': [{'x': cp['x'], 'y': cp['y']} for cp in cross_peaks],
        },
    }
