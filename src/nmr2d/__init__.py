"""nmr2d: generate 2D NMR nmrML files.

Two ways to make a 2D nmrML file that JSpectraViewer (and the jsv2 library) can display:
  - generate_predicted: a predicted 1H-13C HSQC from an assigned 1D 1H and an assigned
    1D 13C nmrML file of the same compound
  - generate_from_dataset: one 2D COSY or HSQC dataset (raw FID or processed spectrum),
    processed with nmrglue

Both return a Result with the nmrML text, a suggested file name and a short summary.
"""
import os
import re
from dataclasses import dataclass, field

from .nmrml_reader import NmrmlError, read_1d_nmrml
from .nmrml_writer import build_2d_nmrml
from .predict import HSQC_DEFAULTS, PredictionError, predict_hsqc
from .process import InputError, process_dataset

__version__ = '0.1.0'
__all__ = ['Result', 'generate_predicted', 'generate_from_dataset', 'read_1d_nmrml', 'predict_hsqc',
           'process_dataset', 'build_2d_nmrml', 'HSQC_DEFAULTS', 'NmrmlError', 'PredictionError', 'InputError']


@dataclass
class Result:
    nmrml: str                 # the 2D nmrML document
    file_name: str             # suggested file name
    summary: str               # one line describing the result
    notes: list = field(default_factory=list)


def base_name(file_name):
    """File name without its extension, made safe for a new file name (as JSpectraViewer does)."""
    stem = re.sub(r'\.[^.]+$', '', os.path.basename(str(file_name)))
    return re.sub(r'[^\w.-]+', '_', stem, flags=re.ASCII) or 'spectrum'


def generate_predicted(proton, carbon, options=None):
    """Predicted 1H-13C HSQC nmrML from 1D 1H and 13C nmrML files (paths, bytes or text)."""
    spectrum = predict_hsqc(read_1d_nmrml(proton), read_1d_nmrml(carbon), options)
    n = len(spectrum['cross_peaks'])
    return Result(nmrml=build_2d_nmrml(spectrum),
                  file_name=base_name(spectrum['name']) + '_predicted_HSQC.nmrML',
                  summary=f'{n} cross peak{"s" if n != 1 else ""}')


def generate_from_dataset(path, original_name=None, **options):
    """2D nmrML from one 2D dataset (file, folder or .zip); options as for process_dataset.
    original_name names the output when path is a temporary copy of an upload."""
    s = process_dataset(path, **options)
    nuclei = s['nuclei']
    if len(nuclei) == 2 and nuclei[0] and nuclei[0] == nuclei[1]:
        experiment = f'{nuclei[0]}-{nuclei[1]} COSY'
    elif len(nuclei) == 2:
        experiment = f'{nuclei[0]}-{nuclei[1]} HSQC'
    else:
        experiment = '2D NMR'
    name = base_name(original_name or path)
    xml = build_2d_nmrml({'x': s['x'], 'y': s['y'], 'z': s['z'], 'id': '2D-' + name, 'name': name,
                          'experiment': f'{experiment} ({s["source"]})'})
    return Result(nmrml=xml, file_name=name + '_2D.nmrML',
                  summary=f'{len(s["x"])} x {len(s["y"])} points, {experiment}', notes=s['notes'])
