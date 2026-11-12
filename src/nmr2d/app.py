"""Streamlit page: the 2D nmrML generator, laid out and styled like JSpectraViewer's page.

Run with:  nmr2d app        (or: streamlit run streamlit_app.py)
"""
import os
import re
import tempfile

import streamlit as st

from nmr2d import InputError, NmrmlError, PredictionError, generate_from_dataset, generate_predicted

# Where the "JSpectraViewer" links point; set JSPECTRAVIEWER_URL to the live site's address
JSPECTRAVIEWER_URL = os.environ.get('JSPECTRAVIEWER_URL', 'https://github.com/JasonZhai-9936/JSV2')

PREDICT = ('Predicted 1H-13C HSQC from an assigned 1D 1H spectrum and an assigned 1D 13C spectrum '
           'of the same compound (.nmrML)')
DATASET = ('One 2D COSY or HSQC dataset: raw FID or processed spectrum (zipped Bruker or Agilent/Varian '
           'experiment folder, NMRPipe .ft2, Sparky .ucsf)')
NOISE_OPTIONS = {'auto': 'Automatic (only for noisy spectra)', 'on': 'Always', 'off': 'Off'}

# JSpectraViewer look: Bootstrap 5.3 colours, pale-yellow navbar and footer, dark-bordered
# card with a highlighted title band
CSS = """
<style>
  :root { --jsv-yellow: #fff3cd; --jsv-amber: #ffc107; --jsv-dark: #212529; --jsv-muted: #6c757d; }
  html, body, [class*="css"], .stApp {
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  header[data-testid="stHeader"] { display: none; }
  #MainMenu, footer { visibility: hidden; }
  .block-container { max-width: 75%; min-width: 320px; padding-top: 5.5rem; padding-bottom: 6rem; }
  @media (max-width: 900px) { .block-container { max-width: 100%; } }

  .jsv-navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; display: flex;
    align-items: center; justify-content: space-between; padding: 0.5rem 1rem;
    background: var(--jsv-yellow); box-shadow: 0 .125rem .25rem rgba(0,0,0,.075); }
  .jsv-navbar .brand { font-size: 1.25rem; font-weight: 500; color: var(--jsv-dark); text-decoration: none;
    padding: 0.375rem 0.75rem; border-right: 1px solid rgba(0,0,0,.15); }
  .jsv-navbar .links { display: flex; gap: 0.5rem; }
  .jsv-navbar .links a { font-size: 1.25rem; font-weight: 500; color: var(--jsv-dark); text-decoration: none;
    padding: 0.375rem 0.75rem; border-radius: 0.375rem; }
  .jsv-navbar .links a:hover { color: #0d6efd; }
  .jsv-navbar .links a.active { background: var(--jsv-amber); box-shadow: 0 .125rem .25rem rgba(0,0,0,.075); }

  .st-key-jsv-card, div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--jsv-dark) !important;
    border-radius: 0.375rem; background: #f8f9fa; box-shadow: 0 .125rem .25rem rgba(0,0,0,.075); }
  .jsv-title { font-size: 1.75rem; font-weight: 600; line-height: 1.2; background: var(--jsv-yellow);
    border: 1px solid #dee2e6; border-radius: 0.375rem; padding: 0.4rem 0.5rem; margin-bottom: 1rem; }
  .jsv-lead { font-size: 1.25rem; font-weight: 300; margin: 0.25rem 0 0.25rem; color: var(--jsv-dark); }
  .jsv-muted { color: var(--jsv-muted); }
  .jsv-status { color: var(--jsv-muted); padding-top: 0.45rem; }
  .jsv-status.error { color: #dc3545; }
  .jsv-notes { color: var(--jsv-muted); font-size: 0.9rem; margin: 0.5rem 0 0 1.2rem; }

  .stButton > button[kind="primary"] { background: #0d6efd; border-color: #0d6efd; }
  .stDownloadButton > button { background: #6c757d; border-color: #6c757d; color: #fff; }
  .stDownloadButton > button:hover { background: #5c636a; border-color: #565e64; color: #fff; }
  .stDownloadButton > button:disabled { background: #6c757d; opacity: 0.65; color: #fff; }

  .jsv-footer { position: fixed; bottom: 0; left: 0; right: 0; background: var(--jsv-yellow);
    padding: 0.75rem 1rem; text-align: right; font-size: 0.9rem; z-index: 999; }
</style>
"""


def navbar():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="jsv-navbar"><a class="brand" href="{JSPECTRAVIEWER_URL}" target="_blank">JSpectraViewer</a>'
        f'<div class="links"><a href="{JSPECTRAVIEWER_URL}" target="_blank">Home</a>'
        f'<a class="active" href="#">2D nmrML Generator</a></div></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="jsv-footer">This project is supported by '
                '<a href="https://www.nih.gov/" target="_blank"><b>NIH</b></a>, '
                '<a href="https://www.nccih.nih.gov/" target="_blank"><b>NIH NCCIH</b></a>, and '
                '<a href="https://ods.od.nih.gov/Funding/Grants__Contracts.aspx" target="_blank"><b>ODH</b></a>. '
                'It is developed by the <a href="https://www.wishartlab.com/" target="_blank"><b>Wishart Research Group</b></a> '
                'at the <a href="https://www.ualberta.ca/" target="_blank"><b>University of Alberta</b></a>.</div>',
                unsafe_allow_html=True)


def lead(text):
    st.markdown(f'<p class="jsv-lead">{text}</p>', unsafe_allow_html=True)


def run_predicted(proton_file, carbon_file):
    if proton_file is None or carbon_file is None:
        raise PredictionError('Choose both a 1D 1H and a 1D 13C nmrML file.')
    return generate_predicted(proton_file.getvalue(), carbon_file.getvalue())


def run_dataset(upload, magnitude, reverse_f1, noise):
    if upload is None:
        raise InputError('Choose a 2D dataset file.')
    ext = re.sub(r'[^.a-z0-9]', '', os.path.splitext(upload.name)[1].lower())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'input' + ext)
        with open(path, 'wb') as f:
            f.write(upload.getbuffer())
        return generate_from_dataset(path, original_name=upload.name, magnitude=magnitude,
                                     reverse_f1=reverse_f1, noise=noise)


def main():
    st.set_page_config(page_title='2D nmrML Generator', page_icon='🧪', layout='wide')
    navbar()

    with st.container(border=True, key='jsv-card'):
        st.markdown('<div class="jsv-title">Generate a 2D NMR nmrML file:</div>', unsafe_allow_html=True)
        st.markdown('<p class="jsv-muted">Create a 2D nmrML file that JSpectraViewer (and the jsv2 library) can display, '
                    'either by predicting a 1H-13C HSQC spectrum from assigned 1D spectra, or by processing a 2D COSY '
                    'or HSQC dataset with <a href="https://www.nmrglue.com/" target="_blank">nmrglue</a>.</p>',
                    unsafe_allow_html=True)

        lead('Step 1: Input:')
        mode = st.radio('Input', (PREDICT, DATASET), label_visibility='collapsed', key='mode')

        if mode == PREDICT:
            lead('Step 2: Upload the 1D spectra:')
            left, right = st.columns(2)
            proton = left.file_uploader('1D 1H spectrum (.nmrML with assignments)', type=['nmrml', 'xml'], key='h')
            carbon = right.file_uploader('1D 13C spectrum (.nmrML with assignments)', type=['nmrml', 'xml'], key='c')
        else:
            lead('Step 2: Upload the 2D dataset:')
            dataset = st.file_uploader('2D dataset', type=['zip', 'ft2', 'ft', 'pipe', 'ucsf'],
                                       label_visibility='collapsed', key='dataset')
            opt1, opt2 = st.columns(2)
            magnitude = opt1.checkbox('Magnitude mode (recommended for raw FIDs and COSY)', value=True, key='magnitude',
                                      help='Untick for a phase-sensitive spectrum.')
            reverse_f1 = opt2.checkbox('Mirror the indirect (F1) dimension', key='reverse_f1',
                                       help='For data whose F1 sign convention is the opposite '
                                            '(the spectrum looks flipped top to bottom).')
            noise_col, _ = st.columns([2, 3])
            noise = noise_col.selectbox('Noise suppression:', list(NOISE_OPTIONS), format_func=NOISE_OPTIONS.get,
                                        key='noise', help='Automatic removes the noise floor only from noisy spectra, '
                                        'so contours are drawn around the signals; clean spectra are not changed.')

        result = st.session_state.get('result')
        if result and result['mode'] != mode:
            result = None
        generate_col, download_col, status_col = st.columns([1, 1.4, 5])
        clicked = generate_col.button('⚙ Generate', type='primary', use_container_width=True)
        if clicked:
            try:
                with st.spinner('Processing the 2D dataset...' if mode == DATASET else 'Predicting the HSQC spectrum...'):
                    out = (run_predicted(proton, carbon) if mode == PREDICT
                           else run_dataset(dataset, magnitude, reverse_f1, noise))
                result = {'mode': mode, 'ok': True, 'value': out}
            except (InputError, NmrmlError, PredictionError, ValueError, OSError) as e:
                result = {'mode': mode, 'ok': False, 'error': str(e)}
            st.session_state['result'] = result

        ok = bool(result and result['ok'])
        download_col.download_button('⬇ Download nmrML', data=result['value'].nmrml if ok else '',
                                     file_name=result['value'].file_name if ok else '2D.nmrML',
                                     mime='application/xml', disabled=not ok, use_container_width=True)
        if result:
            if ok:
                v = result['value']
                status_col.markdown(f'<div class="jsv-status">Generated {v.file_name} ({v.summary}).</div>',
                                    unsafe_allow_html=True)
                if v.notes:
                    st.markdown('<ul class="jsv-notes">' + ''.join(f'<li>{n}</li>' for n in v.notes) + '</ul>',
                                unsafe_allow_html=True)
            else:
                status_col.markdown(f'<div class="jsv-status error">{result["error"]}</div>', unsafe_allow_html=True)

        st.markdown('<p class="jsv-muted" style="margin-top:1rem;font-size:0.9rem;">Open the downloaded file in '
                    f'<a href="{JSPECTRAVIEWER_URL}" target="_blank">JSpectraViewer</a> (2D NMR Spectrum) to view it.</p>',
                    unsafe_allow_html=True)
    footer()


if __name__ == '__main__':
    main()
