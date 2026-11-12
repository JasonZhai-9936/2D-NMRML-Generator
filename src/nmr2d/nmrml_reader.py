"""Read the parts of a 1D nmrML file that the HSQC prediction needs.

Follows jsv2's DataParser (JSpectraViewer) so both tools read files the same way:
- spectrum: the first <spectrumDataArray>, base64 64-bit floats read as interleaved
  (ppm, intensity) pairs, which is how HMDB/NP-MRD 1D nmrML files store them
- assignments: the multiplets of the first <atomAssignment>, each with its peak centres
  and atom references ("a13" -> "13")
- structure: the first <structure>, its atoms (id digits, element, coordinates) and bonds
- compound name: <chemicalCompound><identifierList><identifier name="...">

Differences from jsv2, which only matter for files jsv2 cannot read: zlib-compressed
spectra (compressed="true") are decompressed, and a multiplet without peaks is skipped
instead of discarding all assignments.
"""
import base64
import re
import zlib
import xml.etree.ElementTree as ET

import numpy as np


class NmrmlError(ValueError):
    pass


_FLOAT_PREFIX = re.compile(r'^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Infinity)')
_INT_PREFIX = re.compile(r'^\s*([+-]?\d+)')


def js_parse_float(text):
    """JavaScript parseFloat: the leading number of a string, NaN if there is none."""
    m = _FLOAT_PREFIX.match(str(text)) if text is not None else None
    if not m:
        return float('nan')
    s = m.group(1)
    return float(s.replace('Infinity', 'inf'))


def js_parse_int(text):
    m = _INT_PREFIX.match(str(text)) if text is not None else None
    return int(m.group(1)) if m else None


def _local(tag):
    """Element name without its XML namespace."""
    return tag.rsplit('}', 1)[-1]


def _first(root, name):
    for el in root.iter():
        if _local(el.tag) == name:
            return el
    return None


def _children(el, name):
    return [c for c in el if _local(c.tag) == name] if el is not None else []


def _digits(text):
    return re.sub(r'\D', '', str(text))


def _section(text, tag, greedy=False):
    """Parse the first <tag ...>...</tag> of the document on its own.

    Many real nmrML files (e.g. HMDB's) are not well-formed XML as a whole (unclosed <cv>
    tags, broken attributes), so like jsv2 only the needed sections are parsed. greedy
    matches up to the last closing tag, as jsv2 does for <atomAssignment>."""
    pattern = rf'<{tag}\b[^>]*?(?:/>|>.*{"" if greedy else "?"}</{tag}\s*>)'
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    try:
        return ET.fromstring(_strip_prefixes(m.group(0)))
    except ET.ParseError as e:
        raise NmrmlError(f'Could not read the <{tag}> section of the nmrML file: {e}') from e


def _strip_prefixes(xml):
    """Drop namespace prefixes (nmrml:atom -> atom); a section has no xmlns declarations."""
    return re.sub(r'(</?)[A-Za-z_][\w.-]*:', r'\1', xml)


def decode_spectrum(element):
    """Decode a <spectrumDataArray> into (x, y): interleaved little-endian float64 pairs."""
    text = re.sub(r'\s+', '', element.text or '')
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError as e:
        raise NmrmlError(f'Invalid base64 spectrum data: {e}') from e
    if str(element.get('compressed', '')).lower() == 'true':
        try:
            raw = zlib.decompress(raw)
        except zlib.error as e:
            raise NmrmlError(f'Could not decompress the spectrum data: {e}') from e
    if len(raw) % 8:
        raise NmrmlError('Spectrum data is not a whole number of 64-bit floats.')
    values = np.frombuffer(raw, dtype='<f8')
    return values[0::2].copy(), values[1::2].copy()


def read_structure(text):
    """Atoms and bonds of the first <structure>, or None. Atoms are keyed by the digits of
    their id, in the order JavaScript would list those keys (integer keys ascending)."""
    structure = _section(text, 'structure')
    if structure is None:
        return None
    atom_list = _children(structure, 'atomList')
    bond_list = _children(structure, 'bondList')
    atoms_el = _children(atom_list[0], 'atom') if atom_list else []
    bonds_el = _children(bond_list[0], 'bond') if bond_list else []
    # jsv2 requires at least two atoms and two bonds (its XML parser only makes lists of 2+)
    if len(atoms_el) < 2 or len(bonds_el) < 2:
        return None
    atoms = {}
    for a in atoms_el:
        number = _digits(a.get('id', ''))
        coord = lambda k: (lambda v: 0.0 if v != v or v == 0 else v)(js_parse_float(a.get(k)))
        atoms[number] = {'id': number, 'element': a.get('elementType'), 'x': coord('x'), 'y': coord('y'), 'z': coord('z')}
    bonds = []
    for b in bonds_el:
        refs = str(b.get('atomRefs', '')).strip().split()
        a1, a2 = (_digits(r) for r in (refs + ['', ''])[:2])
        order = js_parse_int(b.get('order'))
        bonds.append({'a1': a1, 'a2': a2, 'order': order if order else 1})
    return {'atoms': _js_key_order(atoms), 'bonds': bonds}


def _js_key_order(d):
    """Reorder a dict like a JavaScript object: canonical integer keys ascending first."""
    is_index = lambda k: k.isdigit() and (k == '0' or not k.startswith('0'))
    keys = sorted((k for k in d if is_index(k)), key=int) + [k for k in d if not is_index(k)]
    return {k: d[k] for k in keys}


def read_assignment(text):
    """Name, structure and assigned clusters of the <atomAssignment>, or None."""
    assignment = _section(text, 'atomAssignment', greedy=True)
    if assignment is None:
        return None
    structure = read_structure(text) if _first(assignment, 'structure') is not None else None

    name = 'Compound'
    compound = _children(assignment, 'chemicalCompound')
    identifier_lists = _children(compound[0], 'identifierList') if compound else []
    identifiers = _children(identifier_lists[0], 'identifier') if identifier_lists else []
    if len(identifiers) == 1:
        name = identifiers[0].get('name')          # may be None, as in jsv2
    elif len(identifiers) > 1:
        name = None

    clusters = []
    lists = _children(assignment, 'atomAssignmentList')
    for multiplet in (_children(lists[0], 'multiplet') if lists else []):
        peak_lists = _children(multiplet, 'peakList')
        peaks = [js_parse_float(p.get('center')) for p in (_children(peak_lists[0], 'peak') if peak_lists else [])]
        atoms = _children(multiplet, 'atoms')
        refs = atoms[0].get('atomRefs') if atoms else None
        atom_refs = [_digits(r) for r in refs.strip().split()] if refs else []
        if peaks:
            clusters.append({'peaks': peaks, 'atom_refs': atom_refs})
    return {'name': name, 'structure': structure, 'clusters': clusters}


def read_1d_nmrml(source):
    """Read a 1D nmrML file (path, bytes or text).

    Returns {'x': ppm values, 'y': intensities, 'assignment': {...} or None} where
    assignment = {'name', 'structure': {'atoms', 'bonds'} or None,
                  'clusters': [{'peaks': [centres], 'atom_refs': ['13', ...]}]}.
    """
    if isinstance(source, (bytes, bytearray)):
        text = bytes(source).decode('utf-8', errors='replace')
    elif isinstance(source, str) and source.lstrip().startswith('<'):
        text = source
    else:
        with open(source, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    data = _section(text, 'spectrumDataArray')
    if data is None:
        raise NmrmlError("No data found with tag 'spectrumDataArray'. Is this a 1D nmrML file?")
    x, y = decode_spectrum(data)
    return {'x': x, 'y': y, 'assignment': read_assignment(text)}
