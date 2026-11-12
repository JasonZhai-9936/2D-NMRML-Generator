# 2D nmrML Generator

Generate 2D NMR spectra as [nmrML](https://nmrml.org) files that [JSpectraViewer](https://github.com/JasonZhai-9936/JSV2) (and the jsv2 library) can display. This is the 2D nmrML Generator from the JSpectraViewer website as a standalone tool, with a web page that looks like the website's, a command line and a Python API.

Two ways to make a file:

| Input | Output |
|---|---|
| An assigned 1D ¹H nmrML file and an assigned 1D ¹³C nmrML file of the same compound | A **predicted ¹H-¹³C HSQC**: each assigned carbon with bonded hydrogens gives one cross peak at the shifts of those hydrogens and that carbon. Quaternary carbons give none. The structure and the cross-peak assignments are written into the file. |
| One 2D COSY or HSQC dataset, raw FID or processed spectrum: a zipped Bruker or Agilent/Varian experiment folder, an NMRPipe `.ft2` or a Sparky `.ucsf` | The spectrum processed with [nmrglue](https://www.nmrglue.com/): Fourier transform, phasing, solvent filter and, for noisy spectra, noise suppression. |

## Install

Python 3.9 or newer.

```bash
git clone <this repository> nmrml-2d-generator
cd nmrml-2d-generator
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Web page

```bash
.venv/bin/nmr2d app
```

This opens the page in your browser (at http://localhost:8501). Choose the input, upload the files, click **Generate**, then **Download nmrML**. Open the downloaded file in JSpectraViewer as a "2D NMR Spectrum" to view it.

To run it from the repository folder instead: `.venv/bin/streamlit run streamlit_app.py`. The page can also be hosted on [Streamlit Community Cloud](https://streamlit.io/cloud) from this repository (main file `streamlit_app.py`). Set the environment variable `JSPECTRAVIEWER_URL` to change where the page's JSpectraViewer links point.

## Command line

```bash
# Predicted HSQC from assigned 1D 1H and 13C nmrML files
nmr2d predict HMDB0000001_1H.nmrML HMDB0000001_13C.nmrML -o out/

# A 2D dataset (zip, experiment folder, .ft2 or .ucsf)
nmr2d dataset hsqc_experiment.zip -o hsqc.nmrML
```

Options for `dataset`:

| Option | Effect |
|---|---|
| `--phase-sensitive` | Phase-sensitive spectrum instead of magnitude mode. Magnitude mode is the default because it is the most reliable for raw FIDs and COSY. |
| `--reverse-f1` | Mirror the indirect (F1) dimension, for data whose F1 sign convention is the opposite of what was assumed. |
| `--noise auto\|on\|off` | Noise suppression (see below). The default is `auto`. |
| `--no-solvent-filter` | Keep the solvent signal at the ¹H carrier. |

## Python

```python
from nmr2d import generate_predicted, generate_from_dataset

result = generate_predicted('HMDB0000001_1H.nmrML', 'HMDB0000001_13C.nmrML')
open(result.file_name, 'w').write(result.nmrml)          # also: result.summary

result = generate_from_dataset('hsqc_experiment.zip', magnitude=True, noise='auto')
print(result.summary, result.notes)
```

The lower-level pieces are importable too: `read_1d_nmrml`, `predict_hsqc`, `process_dataset` and `build_2d_nmrml`.

## How the output relates to the JSpectraViewer website

The same files give the same results as the website:

- **Predicted HSQC:** a port of jsv2's `predict_hsqc`. On the real HMDB0000001 ¹H and ¹³C files it gives jsv2's 6 cross peaks, axes and structure exactly. The nmrML text is identical apart from 85 of the 524,288 intensities, which differ in their last binary digit (relative difference 1.5×10⁻¹⁶) because JavaScript's `Math.exp` rounds a few values differently.
- **2D datasets:** the same processing code as the website, and the same nmrML writer, which reproduces jsv2's output byte for byte, including JavaScript's number formatting.
- **Reading 1D nmrML:** follows jsv2's `DataParser`. Spectra are interleaved (ppm, intensity) pairs, as in HMDB and NP-MRD files. Only the needed sections are parsed, because real files are often not well-formed XML as a whole (for example an unclosed `<cv>` tag). Unlike jsv2, zlib-compressed spectra are also read.

## Notes on 2D processing

- **Bruker:** processed data (`pdata/N/2rr`) is used when present, with TopSpin's referenced axes. Otherwise the raw `ser` is processed. Sign conventions were checked on a real Echo-Antiecho HSQC (HMDB0000001), where the six known cross peaks are the strongest signals. Raw uploads have no TopSpin referencing (SR), so their axes can be off by about 0.1 ppm (¹H) and 2 ppm (¹³C).
- **Agilent/Varian:** axes come from `procpar` (checked on a real HMDB 1D FID). FID pairs are combined with the file's own `f1coef` recipe. The direction of the indirect axis has not been checked on a real 2D file; if a spectrum looks mirrored top to bottom, use `--reverse-f1`.
- **Noise suppression (`auto`):** JSpectraViewer draws its lowest contour at one standard deviation of the spectrum, which on a noisy spectrum falls inside the noise and covers the plot in contours. The noise level is estimated robustly (median and median absolute deviation). Only when that lowest contour would be less than 5 noise standard deviations above the baseline is that 5-sigma floor removed, keeping each point's sign. Noise drops to zero, peaks keep their shape, and clean spectra are returned unchanged. Real peaks weaker than about 5× the noise are removed too; use `--noise off` if a faint peak matters.


## Citation

If you use this tool, please cite JSpectraViewer:

> JSpectraViewer: A web-based interactive visualization tool for metabolomics research.
> Sukanta Saha, Jason Zhai, Vasuk Gautam, Harrison Peters, Jason Grant, Mark Berjanskii, Eponine Oler, Krishna M. Kovur, Prashanthi Kovur, David S. Wishart.
