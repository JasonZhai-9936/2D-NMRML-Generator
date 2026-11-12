"""Process one 2D NMR dataset (raw FID or processed spectrum) with nmrglue.

Accepted input: a file, or a directory or .zip holding one dataset:
  - Bruker experiment folder: processed data (pdata/N/2rr) is used when present,
    otherwise the raw ser file is processed (FnMODE QF, States, States-TPPI, TPPI,
    Echo-Antiecho)
  - Agilent/Varian folder (fid + procpar, any letter case), raw
  - NMRPipe processed spectrum (.ft2 / .ft / .pipe)
  - Sparky spectrum (.ucsf)

Raw data: squared sine-bell apodisation, zero filling, Fourier transform and automatic
phasing in both dimensions; a solvent filter at the 1H carrier; optional magnitude mode.
Sign conventions were checked on real HMDB data (Bruker Echo-Antiecho HSQC, Varian 1D).
"""
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import nmrglue as ng

# Bruker FnMODE values (acqu2s) for the indirect dimension acquisition mode
FNMODE = {0: 'undefined', 1: 'QF', 2: 'QSEQ', 3: 'TPPI', 4: 'States', 5: 'States-TPPI', 6: 'Echo-Antiecho'}


MAX_UNZIPPED_BYTES = 2 * 1024 ** 3

# Facts about the input, reported in error messages to help diagnose server-side failures
DIAGNOSTICS = {}


class InputError(Exception):
    pass


# ---------------------------------------------------------------------------
# Locating the dataset
# ---------------------------------------------------------------------------

def unpack(path, workdir):
    """Return a directory or file to read: zips are extracted into workdir."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            if sum(info.file_size for info in zf.infolist()) > MAX_UNZIPPED_BYTES:
                raise InputError('The zip file is too large when extracted (limit 2 GB).')
            zf.extractall(workdir)
        return workdir
    return path


def find_dataset(root):
    """Return (kind, path) for the first recognised dataset under root."""
    if os.path.isfile(root):
        ext = os.path.splitext(root)[1].lower()
        if ext in ('.ft2', '.ft', '.pipe', '.fid'):
            return 'pipe', root
        if ext == '.ucsf':
            return 'sparky', root
        raise InputError('Unrecognised file. Upload a zipped Bruker/Varian experiment folder, '
                         'an NMRPipe (.ft2) or a Sparky (.ucsf) spectrum.')
    for dirpath, dirnames, filenames in sorted(os.walk(root)):
        dirnames.sort()
        if 'acqus' in filenames and ('ser' in filenames or os.path.isdir(os.path.join(dirpath, 'pdata'))):
            return 'bruker', dirpath
        lower = {f.lower() for f in filenames}
        if 'procpar' in lower and 'fid' in lower:
            return 'varian', dirpath
        for f in sorted(filenames):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.ft2', '.pipe'):
                return 'pipe', os.path.join(dirpath, f)
            if ext == '.ucsf':
                return 'sparky', os.path.join(dirpath, f)
    raise InputError('No 2D NMR dataset found (looked for Bruker acqus+ser/pdata, Varian procpar+fid, '
                     'NMRPipe .ft2 or Sparky .ucsf).')


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def process_dim(data, phase=None, autophase=False):
    """Apodize (squared sine bell), zero-fill and Fourier transform along the last axis."""
    data = ng.proc_base.sp(data, off=0.5, end=1.0, pow=2)
    data = ng.proc_base.zf_size(data, next_pow2(data.shape[-1]) * 2)
    data = ng.proc_base.fft(data)
    if phase is not None:
        data = ng.proc_base.ps(data, p0=phase[0], p1=phase[1])
    elif autophase:
        # Phase from the strongest row, applied to all rows
        row = data[np.argmax(np.abs(data).max(axis=1))]
        _, (p0, p1) = ng.proc_autophase.autops(row, 'acme', return_phases=True, disp=False)
        data = ng.proc_base.ps(data, p0=p0, p1=p1)
    return data


def indirect_to_complex(data, mode):
    """Combine interleaved F1 rows into complex F1 points according to the acquisition mode.
    data: rows = F1 increments, columns = F2 frequency (complex, phased)."""
    if mode in ('States', 'States-TPPI'):
        f1 = data[0::2].real + 1j * data[1::2].real
        if mode == 'States-TPPI':
            f1[1::2] *= -1
        return f1
    if mode == 'Echo-Antiecho':
        # Verified on a real Bruker hsqcetgpsisp2.2 dataset (HMDB0000001, after the F2 reversal
        # in read_bruker): the known cross peaks stand out ~1000x over noise with this
        # combination and at most ~20x with the other ones
        echo, antiecho = data[0::2], data[1::2]
        return (echo + antiecho).imag + 1j * (echo - antiecho).real
    if mode == 'TPPI':
        return data.real.astype(complex)
    return data  # QF / QSEQ / undefined: rows already complex in F1


def process_raw(data, mode, f2_phase, magnitude):
    """Process raw 2D time-domain data (rows = F1 increments) into a real spectrum."""
    data = process_dim(data, phase=f2_phase, autophase=(f2_phase is None and not magnitude))
    data = indirect_to_complex(data, mode)
    data = process_dim(data.T, autophase=not magnitude).T  # F1 (transpose so it is the last axis)
    if magnitude or mode in ('QF',):
        return np.abs(data)
    return data.real


def ppm_axes_from_udic(udic, ny, nx):
    """ppm scales for F1 (dim 0) and F2 (dim 1) of a frequency-domain array."""
    for dim in (0, 1):
        # nmrglue fills in 999.99 when it cannot find the spectrometer parameters
        if udic[dim].get('obs') in (None, 999.99) or udic[dim].get('sw') in (None, 999.99):
            axes = {d: {k: udic[d].get(k) for k in ('obs', 'sw', 'car', 'label')} for d in (0, 1)}
            raise InputError('Missing spectrometer frequency or spectral width for the '
                             f'{"indirect" if dim == 0 else "direct"} dimension, so ppm axes cannot be computed. '
                             f'[diagnostics: script v4-varian-procpar, nmrglue {ng.__version__}, '
                             f'numpy {np.__version__}, python {sys.version.split()[0]}, axes {axes}, {DIAGNOSTICS}]')
    scales = []
    for dim, size in ((0, ny), (1, nx)):
        udic[dim].update(size=size, complex=False, time=False, freq=True)
        scales.append(ng.fileiobase.uc_from_udic(udic, dim).ppm_scale())
    return scales[0], scales[1]


def labels(udic):
    return [str(udic[1].get('label') or 'F2'), str(udic[0].get('label') or 'F1')]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def fill_axes_from_acqus(dic, udic):
    """Fill spectrometer frequency, spectral width, carrier and nucleus from the Bruker
    acquisition files wherever nmrglue left them unset (999.99 / missing)."""
    missing = lambda v: v in (None, 999.99, '') or (isinstance(v, float) and np.isnan(v))
    for dim, name in ((0, 'acqu2s'), (1, 'acqus')):
        acq = dic.get(name) or {}
        if not acq:
            continue
        if missing(udic[dim].get('obs')) and acq.get('SFO1'):
            udic[dim]['obs'] = float(acq['SFO1'])
        if missing(udic[dim].get('sw')) and acq.get('SW_h'):
            udic[dim]['sw'] = float(acq['SW_h'])
        if missing(udic[dim].get('car')) and acq.get('O1') is not None:
            udic[dim]['car'] = float(acq['O1'])      # carrier offset in Hz
        label = str(udic[dim].get('label') or '')
        if (not label or label in ('X', 'Y', 'F1', 'F2')) and acq.get('NUC1'):
            udic[dim]['label'] = str(acq['NUC1']).strip('<>')


def processed_ppm_axis(dic, procs_name, size):
    """ppm axis of one dimension of TopSpin processed data, from procs/proc2s."""
    procs = dic.get(procs_name) or {}
    DIAGNOSTICS[procs_name] = {k: procs.get(k) for k in ('OFFSET', 'SW_p', 'SF', 'SI')}
    try:
        offset, sw_hz, sf = float(procs['OFFSET']), float(procs['SW_p']), float(procs['SF'])
    except (KeyError, TypeError, ValueError):
        raise InputError(f'Processed Bruker data is missing OFFSET/SW_p/SF in {procs_name}, '
                         f'so ppm axes cannot be computed. [diagnostics: {DIAGNOSTICS}]')
    if sf <= 0 or sw_hz <= 0:
        raise InputError(f'Processed Bruker data has invalid SW_p/SF in {procs_name}. [diagnostics: {DIAGNOSTICS}]')
    return offset - np.arange(size) * (sw_hz / sf) / size


def read_bruker(path, magnitude, notes, solvent_filter=True):
    pdata_root = os.path.join(path, 'pdata')
    pdata = None
    if os.path.isdir(pdata_root):
        for name in sorted(os.listdir(pdata_root), key=lambda s: (not s.isdigit(), s.zfill(4))):
            if os.path.exists(os.path.join(pdata_root, name, '2rr')):
                pdata = os.path.join(pdata_root, name)
                break
    if pdata:
        dic, data = ng.bruker.read_pdata(pdata, scale_data=True)
        if data.ndim != 2:
            raise InputError(f'Expected 2D processed data, found {data.ndim}D.')
        # TopSpin's processed axes: OFFSET = ppm of the first point, SW_p = width in Hz,
        # SF = spectrometer frequency in MHz (these include the SR referencing)
        y = processed_ppm_axis(dic, 'proc2s', data.shape[0])
        x = processed_ppm_axis(dic, 'procs', data.shape[1])
        nuclei = []
        for name, default in (('acqus', 'F2'), ('acqu2s', 'F1')):
            acq_file = os.path.join(path, name)
            acq = ng.bruker.read_jcamp(acq_file) if os.path.exists(acq_file) else {}
            nuclei.append(str(acq.get('NUC1') or default).strip('<>'))
        notes.append(f'Processed Bruker data ({os.path.relpath(pdata, os.path.dirname(path))}) used as is.')
        z = np.asarray(data, dtype=float)
        return (np.abs(z) if magnitude else z), x, y, nuclei, 'bruker-processed'

    if not os.path.exists(os.path.join(path, 'ser')):
        raise InputError('Bruker folder has neither processed 2D data (pdata/N/2rr) nor a raw ser file.')
    # Raw-data-only uploads often have an empty pdata folder: only read procs when it exists
    has_procs = os.path.isdir(pdata_root) and any(
        os.path.exists(os.path.join(pdata_root, name, 'procs')) for name in os.listdir(pdata_root))
    dic, data = ng.bruker.read(path, read_procs=has_procs)
    if data.ndim != 2:
        raise InputError(f'Expected 2D data, found {data.ndim}D.')
    # Some nmrglue versions do not load acqu2s (the indirect dimension's parameters)
    for name in ('acqus', 'acqu2s'):
        if name not in dic and os.path.exists(os.path.join(path, name)):
            dic[name] = ng.bruker.read_jcamp(os.path.join(path, name))
    udic = ng.bruker.guess_udic(dic, data)
    DIAGNOSTICS['guessed'] = {d: (udic[d].get('obs'), udic[d].get('sw')) for d in (0, 1)}
    DIAGNOSTICS['files'] = sorted(os.listdir(path))
    DIAGNOSTICS['dic'] = sorted(k for k in dic if not k.startswith('_'))
    DIAGNOSTICS['acqu2s'] = {k: (dic.get('acqu2s') or {}).get(k) for k in ('SFO1', 'SW_h', 'O1', 'NUC1')}
    fill_axes_from_acqus(dic, udic)
    try:
        data = ng.bruker.remove_digital_filter(dic, data)
    except (KeyError, ValueError):
        notes.append('No Bruker digital filter parameters (GRPDLY/DECIM/DSPFVS); filter correction skipped.')
    # Bruker raw data comes out mirrored in F2 with nmrglue's FFT; conjugating the time
    # domain reverses it (checked on real data: 1H signals otherwise mirror about the carrier)
    data = np.conj(data)
    if solvent_filter:
        # Remove the residual water line at the carrier, which otherwise outweighs the cross peaks
        data = ng.proc_bl.sol_gaussian(data, w=32)
        notes.append('Solvent signal at the 1H carrier removed (solvent filter).')
    mode = FNMODE.get(int(dic.get('acqu2s', {}).get('FnMODE', 0)), 'undefined')
    # F2 is phased automatically (TopSpin's PHC0/PHC1 assume its own sign convention)
    spectrum = process_raw(data, mode, None, magnitude)
    y, x = ppm_axes_from_udic(udic, spectrum.shape[0], spectrum.shape[1])
    notes.append(f'Raw Bruker ser processed (F1 mode {mode}{", magnitude" if magnitude or mode == "QF" else ""}).')
    if not has_procs:
        notes.append('No processing parameters (pdata/1/procs), so no TopSpin chemical shift '
                     'referencing (SR): ppm axes come from the acquisition parameters and may be '
                     'offset by about 0.1 ppm.')
    return spectrum, x, y, labels(udic), 'bruker-raw'


def find_file(directory, name):
    """File in directory whose name matches case-insensitively (HMDB's Varian zips use FID/PROCPAR)."""
    for f in os.listdir(directory):
        if f.lower() == name:
            return f
    return None


def varian_nucleus(name):
    """'H1' -> '1H', 'C13' -> '13C'."""
    m = re.match(r'^([A-Za-z]+)(\d+)$', str(name).strip())
    return f'{m.group(2)}{m.group(1).capitalize()}' if m else str(name)


def varian_axes(procpar, udic, notes):
    """Fill the axes from procpar (nmrglue's guess_udic leaves them unset for Varian data).
    Spectrum centre in Hz = sw/2 - rfl + rfp (Varian referencing); checked on a real HMDB
    Varian 1D 1H FID, where the peaks land exactly on the HMDB shifts."""
    val = lambda k, default=None: (procpar.get(k, {}).get('values') or [default])[0]
    num = lambda k, default=0.0: float(val(k, default) or default)
    seq = str(val('seqfil', ''))
    tn, dn = str(val('tn', 'H1')), str(val('dn', 'C13'))
    # Indirect dimension: the decoupler nucleus for heteronuclear experiments, else the observe nucleus
    hetero = bool(re.search(r'hsqc|hmqc|hmbc|hetcor|hetco', seq, re.I))
    f1_obs, f1_nuc = (num('dfrq'), dn) if hetero else (num('sfrq'), tn)
    if not hetero and dn != tn and not re.search(r'cosy|tocsy|noesy|roesy|dqf', seq, re.I):
        notes.append(f'Pulse sequence "{seq}" not recognised: indirect dimension assumed to be {varian_nucleus(tn)}.')
    for dim, obs, sw, rfl, rfp, nuc in (
            (1, num('sfrq'), num('sw'), num('rfl'), num('rfp'), tn),
            (0, f1_obs, num('sw1'), num('rfl1'), num('rfp1'), f1_nuc)):
        udic[dim].update(obs=obs, sw=sw, car=sw / 2 - rfl + rfp, label=varian_nucleus(nuc))
    DIAGNOSTICS['varian'] = {'seqfil': seq, 'tn': tn, 'dn': dn, 'hetero': hetero,
                             'axes': {d: (udic[d]['obs'], udic[d]['sw'], udic[d]['car']) for d in (0, 1)}}


def varian_combine(data, f1coef):
    """Combine each increment's FID pair as the f1coef recipe says (VnmrJ convention): the
    8 numbers are complex multipliers (re, im) of FID 1 and FID 2 for the two outputs,
    which are then processed as a States pair."""
    c = [float(v) for v in f1coef]
    fid1, fid2 = data[0::2], data[1::2]
    out = np.empty_like(data)
    out[0::2] = complex(c[0], c[1]) * fid1 + complex(c[2], c[3]) * fid2
    out[1::2] = complex(c[4], c[5]) * fid1 + complex(c[6], c[7]) * fid2
    return out


def read_varian(path, magnitude, notes, solvent_filter=True):
    DIAGNOSTICS['route'] = 'varian'
    fid_file, procpar_file = find_file(path, 'fid'), find_file(path, 'procpar')
    dic, data = ng.varian.read(path, fid_file=fid_file, procpar_file=procpar_file)
    if data.ndim != 2:
        raise InputError(f'Expected 2D data, found {data.ndim}D (this looks like a 1D experiment).')
    if data.shape[0] % 2:
        raise InputError('Expected an even number of FIDs (two per increment) in the Varian 2D data.')
    procpar = dic.get('procpar', {})
    udic = ng.varian.guess_udic(dic, data)
    varian_axes(procpar, udic, notes)

    f1coef = str((procpar.get('f1coef', {}).get('values') or [''])[0]).split()
    if len(f1coef) == 8:
        data = varian_combine(data, f1coef)
        how = f'f1coef {" ".join(f1coef)}'
    else:
        how = 'States (no f1coef)'
    if solvent_filter:
        data = ng.proc_bl.sol_gaussian(data, w=32)
        notes.append('Solvent signal at the 1H carrier removed (solvent filter).')
    spectrum = process_raw(data, 'States', None, magnitude)
    y, x = ppm_axes_from_udic(udic, spectrum.shape[0], spectrum.shape[1])
    notes.append(f'Raw Agilent/Varian fid processed ({how}{", magnitude" if magnitude else ""}). '
                 'The Varian indirect-dimension sign has not been checked on real 2D data: if the '
                 'spectrum looks mirrored top to bottom, tick "Mirror the indirect (F1) dimension".')
    return spectrum, x, y, [udic[1]['label'], udic[0]['label']], 'varian-raw'


def read_pipe(path, magnitude, notes):
    dic, data = ng.pipe.read(path)
    if data.ndim != 2:
        raise InputError(f'Expected 2D data, found {data.ndim}D.')
    udic = ng.pipe.guess_udic(dic, data)
    if not (udic[0].get('freq') and udic[1].get('freq')):
        raise InputError('NMRPipe file is not fully processed (both dimensions must be frequency domain).')
    data = np.abs(data) if magnitude else np.real(data)
    y = ng.pipe.make_uc(dic, data, dim=0).ppm_scale()
    x = ng.pipe.make_uc(dic, data, dim=1).ppm_scale()
    notes.append('Processed NMRPipe spectrum used as is.')
    return np.asarray(data, dtype=float), x, y, labels(udic), 'nmrpipe'


def read_sparky(path, magnitude, notes):
    dic, data = ng.sparky.read(path)
    if data.ndim != 2:
        raise InputError(f'Expected 2D data, found {data.ndim}D.')
    udic = ng.sparky.guess_udic(dic, data)
    y = ng.sparky.make_uc(dic, data, dim=0).ppm_scale()
    x = ng.sparky.make_uc(dic, data, dim=1).ppm_scale()
    notes.append('Sparky spectrum used as is.')
    return np.asarray(data, dtype=float), x, y, labels(udic), 'sparky'


READERS = {'bruker': read_bruker, 'varian': read_varian, 'pipe': read_pipe, 'sparky': read_sparky}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Noise floor, in robust noise standard deviations above the baseline. Random noise on a
# ~500k-point grid reaches about 4.5 sigma, so contours drawn below this level show noise.
# A spectrum counts as noisy when the 2D viewer's lowest contour (1 standard deviation of
# the grid, its default) falls below the floor. Calibrated on real HMDB HSQCs: clean ones
# have their lowest contour 5.5-13 sigma above the baseline; visibly noisy copies 2 or less.
NOISE_FLOOR_SIGMAS = 5.0


def noise_stats(z):
    """Robust noise level of a 2D spectrum (most points are noise): returns (baseline, sigma),
    from the median and the median absolute deviation of |z|."""
    a = np.abs(z)
    baseline = float(np.median(a))
    sigma = 1.4826 * float(np.median(np.abs(a - baseline)))
    return baseline, sigma


def contour_coverage(z):
    """Fraction of points at or above the 2D viewer's lowest contour (1 standard deviation
    of the grid, its default)."""
    return float(np.mean(np.abs(z) >= np.std(z)))


def contour_margin(z):
    """How far the viewer's lowest contour sits above the baseline, in noise standard deviations."""
    baseline, sigma = noise_stats(z)
    return (float(np.std(z)) - baseline) / sigma if sigma > 0 else float('inf')


def suppress_noise(z, mode, notes):
    """Remove the noise floor from noisy spectra so contours are drawn only around signals.

    mode 'auto' only changes spectra whose lowest viewer contour would fall inside the
    noise (contour_margin below NOISE_FLOOR_SIGMAS), so clean spectra are returned
    unchanged; 'on' always applies it and 'off' never does. The floor (baseline +
    NOISE_FLOOR_SIGMAS noise standard deviations) is subtracted from |z| and clipped at
    zero, keeping each point's sign: noise drops to zero and peaks keep their shape,
    lowered by the floor."""
    if mode == 'off':
        return z
    margin = contour_margin(z)
    DIAGNOSTICS['noise'] = {'margin_sigma': round(margin, 1), 'coverage': round(contour_coverage(z), 3)}
    if mode == 'auto' and margin >= NOISE_FLOOR_SIGMAS:
        return z
    baseline, sigma = noise_stats(z)
    floor = baseline + NOISE_FLOOR_SIGMAS * sigma
    a = np.abs(z)
    if floor <= 0 or floor >= a.max():
        return z
    cleaned = np.sign(z) * np.clip(a - floor, 0, None)
    notes.append(f'Noise suppressed: the lowest contours would have been inside the noise, so a '
                 f'floor of {NOISE_FLOOR_SIGMAS:g} noise standard deviations was removed '
                 f'({np.mean(cleaned != 0):.1%} of points kept).')
    return cleaned


def reduce_grid(z, x, y, max_x, max_y, notes):
    """Shrink the grid to at most max_y x max_x by keeping, in each block, the value
    with the largest magnitude (so peaks are not averaged away)."""
    fy = int(np.ceil(z.shape[0] / max_y))
    fx = int(np.ceil(z.shape[1] / max_x))
    if fy <= 1 and fx <= 1:
        return z, x, y
    ny, nx = z.shape[0] // fy, z.shape[1] // fx
    blocks = z[:ny * fy, :nx * fx].reshape(ny, fy, nx, fx).transpose(0, 2, 1, 3).reshape(ny, nx, fy * fx)
    picked = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=2)[..., None], axis=2)[..., 0]
    notes.append(f'Reduced from {z.shape[1]} x {z.shape[0]} to {nx} x {ny} points.')
    return picked, x[:nx * fx].reshape(nx, fx).mean(axis=1), y[:ny * fy].reshape(ny, fy).mean(axis=1)




def process_dataset(input_path, magnitude=False, reverse_f1=False, noise='auto', solvent_filter=True,
                    max_f2=2048, max_f1=1024):
    """Process a 2D dataset into a spectrum.

    input_path     file, directory or .zip (see the module docstring)
    magnitude      magnitude-mode spectrum (recommended for raw FIDs and COSY)
    reverse_f1     mirror the indirect dimension (data with the opposite F1 sign convention)
    noise          'auto' (remove the noise floor only from noisy spectra), 'on' or 'off'
    solvent_filter remove the water line at the 1H carrier from raw data
    max_f2, max_f1 largest grid kept (larger spectra are reduced, keeping peak maxima)

    Returns {'x': F2 ppm, 'y': F1 ppm, 'z': intensities (len(y) x len(x)),
             'nuclei': [F2, F1], 'source', 'notes'}.
    Raises InputError with a readable message for unusable input.
    """
    if noise not in ('auto', 'on', 'off'):
        raise ValueError("noise must be 'auto', 'on' or 'off'")
    DIAGNOSTICS.clear()
    notes = []
    with tempfile.TemporaryDirectory() as workdir:
        kind, path = find_dataset(unpack(str(input_path), workdir))
        try:
            if kind in ('bruker', 'varian'):
                z, x, y, nuclei, source = READERS[kind](path, magnitude, notes, solvent_filter=solvent_filter)
            else:
                z, x, y, nuclei, source = READERS[kind](path, magnitude, notes)
        except InputError:
            raise
        except Exception as e:     # unexpected reader/processing failure
            raise InputError(f'Could not process the 2D data: {e}') from e
    if reverse_f1:
        z = np.asarray(z)[::-1]
        notes.append('Indirect dimension mirrored.')
    z, x, y = reduce_grid(np.asarray(z, dtype=float), np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                          max_f2, max_f1, notes)
    z = suppress_noise(z, noise, notes)
    return {'x': x, 'y': y, 'z': np.asarray(z), 'nuclei': list(nuclei), 'source': source, 'notes': notes}
