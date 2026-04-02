import numpy as np
import scifor as _scifor
from scidb import BaseVariable, configure_database, for_each
from scidb.exceptions import AmbiguousParamError, AmbiguousVersionError
import tempfile, pathlib

class RawSignal(BaseVariable): pass
class Filtered(BaseVariable): pass
class Intermediate(BaseVariable): pass
class Spikes(BaseVariable): pass

def bandpass(signal, low_hz):
    return signal * low_hz

def smooth(signal, threshold):
    return signal * threshold

def detect_spikes(signal, threshold):
    return (signal > threshold).astype(float)

with tempfile.TemporaryDirectory() as tmp:
    _scifor.set_schema([])
    db = configure_database(pathlib.Path(tmp) / 'test.duckdb', ['subject', 'session'])

    RawSignal.save(np.array([1.0, 2.0, 3.0]), subject='S01', session='1')
    for_each(bandpass, {'signal': RawSignal, 'low_hz': 20}, [Filtered], subject=['S01'], session=['1'])
    for_each(bandpass, {'signal': RawSignal, 'low_hz': 30}, [Filtered], subject=['S01'], session=['1'])

    for_each(smooth, {'signal': Filtered, 'threshold': 0.1}, [Intermediate], subject=['S01'], session=['1'])
    for_each(detect_spikes, {'signal': Intermediate, 'threshold': 0.5}, [Spikes], subject=['S01'], session=['1'])

    # Check what branch_params Spikes records have
    spikes_versions = db.list_versions(Spikes, subject='S01', session='1')
    for v in spikes_versions:
        print('branch_params:', v.get('branch_params'))

    db.close()
    _scifor.set_schema([])