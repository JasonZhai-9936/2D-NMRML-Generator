"""Command line: nmr2d predict | dataset | app"""
import argparse
import os
import subprocess
import sys

from . import InputError, NmrmlError, PredictionError, __version__, generate_from_dataset, generate_predicted


def _write(result, output):
    path = output or result.file_name
    if os.path.isdir(path):
        path = os.path.join(path, result.file_name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(result.nmrml)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='nmr2d', description='Generate 2D NMR nmrML files.')
    parser.add_argument('--version', action='version', version=f'nmr2d {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('predict', help='predicted 1H-13C HSQC from assigned 1D 1H and 13C nmrML files')
    p.add_argument('proton', help='assigned 1D 1H nmrML file')
    p.add_argument('carbon', help='assigned 1D 13C nmrML file')
    p.add_argument('-o', '--output', help='output file or folder (default: <compound>_predicted_HSQC.nmrML)')

    d = sub.add_parser('dataset', help='one 2D COSY or HSQC dataset (raw FID or processed spectrum)')
    d.add_argument('input', help='zipped or unzipped Bruker/Agilent/Varian experiment, NMRPipe .ft2 or Sparky .ucsf')
    d.add_argument('-o', '--output', help='output file or folder (default: <input>_2D.nmrML)')
    d.add_argument('--phase-sensitive', action='store_true',
                   help='phase-sensitive spectrum instead of magnitude mode (the default)')
    d.add_argument('--reverse-f1', action='store_true', help='mirror the indirect (F1) dimension')
    d.add_argument('--noise', choices=('auto', 'on', 'off'), default='auto',
                   help='noise suppression: auto = only for noisy spectra (default)')
    d.add_argument('--no-solvent-filter', action='store_true', help='keep the solvent signal at the 1H carrier')

    sub.add_parser('app', help='open the web page (Streamlit)')

    args = parser.parse_args(argv)
    if args.command == 'app':
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
        # Same settings as .streamlit/config.toml, which Streamlit only reads from the current folder
        settings = ['--theme.base', 'light', '--theme.primaryColor', '#0d6efd', '--theme.backgroundColor', '#f8f9fa',
                    '--theme.secondaryBackgroundColor', '#ffffff', '--theme.textColor', '#212529',
                    '--server.maxUploadSize', '500', '--client.toolbarMode', 'minimal',
                    '--server.address', 'localhost',     # only this computer can open the page
                    '--browser.gatherUsageStats', 'false']
        return subprocess.call([sys.executable, '-m', 'streamlit', 'run', app, *settings])
    try:
        if args.command == 'predict':
            result = generate_predicted(args.proton, args.carbon)
        else:
            result = generate_from_dataset(args.input, magnitude=not args.phase_sensitive, reverse_f1=args.reverse_f1,
                                           noise=args.noise, solvent_filter=not args.no_solvent_filter)
    except (InputError, NmrmlError, PredictionError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    path = _write(result, args.output)
    print(f'Wrote {path} ({result.summary}).')
    for note in result.notes:
        print(f'  - {note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
