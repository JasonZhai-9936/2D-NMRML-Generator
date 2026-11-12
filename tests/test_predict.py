import base64
import zlib

import numpy as np
import pytest

from nmr2d import PredictionError, generate_predicted
from nmr2d.nmrml_reader import NmrmlError, read_1d_nmrml
from nmr2d.predict import predict_hsqc

# Ethanol, numbered like HMDB files: C1 (CH3), C2 (CH2), O3, H4-H6 on C1, H7-H8 on C2, H9 on O3
ATOMS = [(1, 'C'), (2, 'C'), (3, 'O'), (4, 'H'), (5, 'H'), (6, 'H'), (7, 'H'), (8, 'H'), (9, 'H')]
BONDS = [(1, 2), (2, 3), (1, 4), (1, 5), (1, 6), (2, 7), (2, 8), (3, 9)]


def nmrml_1d(multiplets, ppm_range, nucleus='hydrogen atom', compressed=False, atoms=ATOMS, name='Ethanol'):
    """A 1D nmrML in the style of HMDB's files, including their malformed header
    (an unclosed <cv> and a broken <chemicalShiftStandard>, which real files have)."""
    x = np.linspace(ppm_range[0], ppm_range[1], 64)
    raw = np.column_stack([x, np.zeros_like(x)]).ravel().astype('<f8').tobytes()
    if compressed:
        raw = zlib.compress(raw)
    data = base64.b64encode(raw).decode()
    atom_xml = ''.join(f'<atom id="a{i}" elementType="{e}" x="{i * 0.5}" y="0" z="0"/>' for i, e in atoms)
    bond_xml = ''.join(f'<bond atomRefs="a{a} a{b}" order="1"/>' for a, b in BONDS)
    mult_xml = ''.join(
        f'<multiplet center="{c}"><atoms atomRefs="{" ".join(f"a{r}" for r in refs)}"/>'
        f'<peakList>{"".join(f"<peak center={chr(34)}{p}{chr(34)} amplitude={chr(34)}1{chr(34)} width={chr(34)}1{chr(34)}/>" for p in peaks)}</peakList></multiplet>'
        for c, refs, peaks in multiplets)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<nmrML xmlns="http://nmrml.org/schema" version="1.0.rc1">
<cvList>
<cv id="NMRCV" fullName="Nuclear Magnetic Resonance CV" version="1.1.0">
</cvList>
<sampleList><sample><chemicalShiftStandard  name="DSS"  </chemicalShiftStandard></sample></sampleList>
<acquisition><acquisition1D><acquisitionParameterSet><acquisitionNucleus name="{nucleus}"/></acquisitionParameterSet></acquisition1D></acquisition>
<spectrumList><spectrum1D numberOfDataPoints="64">
<spectrumDataArray compressed="{'true' if compressed else 'false'}" encodedLength="{len(data)}" byteFormat="complex128">{data}</spectrumDataArray>
</spectrum1D></spectrumList>
<spectrumAnnotationList><atomAssignment spectrumRef="s1">
<chemicalCompound><identifierList><identifier name="{name}"/></identifierList>
<structure><atomList>{atom_xml}</atomList><bondList>{bond_xml}</bondList></structure></chemicalCompound>
<atomAssignmentList>{mult_xml}</atomAssignmentList>
</atomAssignment></spectrumAnnotationList>
</nmrML>'''


PROTON = nmrml_1d([(1.17, [4, 5, 6], [1.16, 1.17, 1.18]), (3.65, [7, 8], [3.64, 3.66]), (2.61, [9], [2.61])], (10, -0.5))
CARBON = nmrml_1d([(17.9, [1], [17.9]), (58.2, [2], [58.2])], (200, 0), nucleus='carbon atom')


def test_reader_handles_malformed_hmdb_style_files():
    s = read_1d_nmrml(PROTON)
    assert s['x'][0] == 10 and s['x'][-1] == -0.5 and len(s['x']) == 64
    a = s['assignment']
    assert a['name'] == 'Ethanol'
    assert [c['atom_refs'] for c in a['clusters']] == [['4', '5', '6'], ['7', '8'], ['9']]
    assert a['clusters'][0]['peaks'] == [1.16, 1.17, 1.18]
    assert list(a['structure']['atoms']) == [str(i) for i, _ in ATOMS]
    assert a['structure']['bonds'][0] == {'a1': '1', 'a2': '2', 'order': 1}


def test_reader_decompresses_zlib_spectra():
    s = read_1d_nmrml(nmrml_1d([(1.17, [4], [1.17])], (10, -0.5), compressed=True))
    assert s['x'][0] == 10 and len(s['x']) == 64


def test_reader_rejects_non_spectra():
    with pytest.raises(NmrmlError):
        read_1d_nmrml('<nmrML><spectrumList/></nmrML>')


def test_predicted_ethanol_hsqc():
    s = predict_hsqc(read_1d_nmrml(PROTON), read_1d_nmrml(CARBON))
    peaks = [(cp['x']['center'], cp['y']['center'], cp['x']['atom_refs'], cp['y']['atom_refs'], cp['intensity'])
             for cp in s['cross_peaks']]
    # The OH (on oxygen) gives no cross peak; the CH3 is three times as intense as one H
    assert peaks == [(1.17, 17.9, ['H4', 'H5', 'H6'], ['C1'], 3), (3.65, 58.2, ['H7', 'H8'], ['C2'], 2)]
    assert s['z'].shape == (512, 1024)
    x, y, z = s['x'], s['y'], s['z']
    for h, c, *_ in peaks:              # the grid maximum near each cross peak sits on it
        rows, cols = np.abs(y - c) < 2, np.abs(x - h) < 0.1
        sub = z[np.ix_(rows, cols)]
        r, k = np.unravel_index(sub.argmax(), sub.shape)
        assert abs(y[rows][r] - c) < 0.5 and abs(x[cols][k] - h) < 0.02
    assert x[0] > x[-1] and y[0] > y[-1]          # high to low ppm


def test_quaternary_carbon_gives_no_cross_peak():
    carbon = nmrml_1d([(17.9, [1], [17.9]), (58.2, [2], [58.2]), (170.0, [3], [170.0])], (200, 0))
    s = predict_hsqc(read_1d_nmrml(PROTON), read_1d_nmrml(carbon))
    assert len(s['cross_peaks']) == 2           # atom 3 is oxygen: ignored


def test_mismatched_structures_rejected():
    other = nmrml_1d([(17.9, [1], [17.9])], (200, 0), atoms=[(1, 'C'), (2, 'N')] + ATOMS[2:])
    with pytest.raises(PredictionError, match='different structures'):
        predict_hsqc(read_1d_nmrml(PROTON), read_1d_nmrml(other))


def test_missing_assignments_rejected():
    plain = '<nmrML><spectrumList><spectrum1D><spectrumDataArray>' + base64.b64encode(
        np.array([1.0, 0.0, 0.0, 0.0]).astype('<f8').tobytes()).decode() + '</spectrumDataArray></spectrum1D></spectrumList></nmrML>'
    with pytest.raises(PredictionError, match='assigned'):
        predict_hsqc(read_1d_nmrml(plain), read_1d_nmrml(CARBON))


def test_generate_predicted_result():
    result = generate_predicted(PROTON.encode(), CARBON.encode())
    assert result.file_name == 'Ethanol_predicted_HSQC.nmrML'
    assert result.summary == '2 cross peaks'
    assert '<multiplet center="17.9"><atoms atomRefs="C1"/></multiplet>' in result.nmrml


# --- Real data: must match what jsv2 (the JSpectraViewer website) produces ---------------

# jsv2's predict_hsqc on the same two HMDB0000001 files (recorded from the JavaScript)
JSV2_CROSS_PEAKS = [
    (3.68, ['H13', 'H14', 'H15'], 39.9, ['C1'], 3), (7.0, ['H16'], 151.579, ['C3'], 1),
    (3.16, ['H17', 'H18'], 30.49, ['C6'], 2), (3.065, ['H17', 'H18'], 30.49, ['C6'], 2),
    (3.96, ['H19'], 55.051, ['C7'], 1), (7.67, ['H23'], 126.26, ['C12'], 1),
]
JSV2_AXES = {'x0': 10.80103, 'x1': -1.196961, 'y0': 199.993896484375, 'y1': 0.00457763671875}


@pytest.mark.network
def test_real_hmdb_matches_jsv2(real_file):
    s = predict_hsqc(read_1d_nmrml(real_file('proton')), read_1d_nmrml(real_file('carbon')))
    got = [(cp['x']['center'], cp['x']['atom_refs'], cp['y']['center'], cp['y']['atom_refs'], cp['intensity'])
           for cp in s['cross_peaks']]
    assert got == JSV2_CROSS_PEAKS
    assert s['name'] == 'HMDB0000001'
    assert (s['x'][0], s['x'][-1], s['y'][0], s['y'][-1]) == tuple(JSV2_AXES.values())
    assert len(s['assignment']['atoms']) == 23 and len(s['assignment']['bonds']) == 23
