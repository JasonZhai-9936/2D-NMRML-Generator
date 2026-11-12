import os
import urllib.request

import pytest

# Real HMDB0000001 (1-methylhistidine) files on MolDB, used by the tests marked "network"
MOLDB = 'https://moldb.wishartlab.com/system/documents/files'
REAL_FILES = {
    'proton': f'{MOLDB}/005/590/893/original/HMDB0000001_nmroned_1022_from_nmrML.nmrML?1648141054',
    'carbon': f'{MOLDB}/002/773/747/original/100_13C_for_JSviewer_HMDB0000001.nmrML?1631592744',
    'hsqc': f'{MOLDB}/000/029/414/original/041031_P00_04_IS_2DHSQC_67420121204-90466-16ua6s0.zip?1354661701',
}
CACHE = os.path.join(os.path.dirname(__file__), '.cache')


@pytest.fixture(scope='session')
def real_file():
    """Download (once) and return the path of a real HMDB file; skip the test when offline."""
    def get(key):
        os.makedirs(CACHE, exist_ok=True)
        ext = '.zip' if key == 'hsqc' else '.nmrML'
        path = os.path.join(CACHE, key + ext)
        if not os.path.exists(path):
            try:
                req = urllib.request.Request(REAL_FILES[key], headers={'User-Agent': 'nmrml-2d-generator tests'})
                with urllib.request.urlopen(req, timeout=60) as r, open(path + '.part', 'wb') as f:
                    f.write(r.read())
                os.replace(path + '.part', path)
            except OSError as e:
                pytest.skip(f'could not download test data ({e})')
        return path
    return get
