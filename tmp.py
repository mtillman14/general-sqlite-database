"""Verification script for branch_params implementation (Items 2 + 3).

Run with: python tmp.py
"""

import numpy as np
import sys
sys.path.insert(0, "scidb/src")
sys.path.insert(0, "scilineage/src")

from scidb import configure_database, BaseVariable, for_each
from scidb.exceptions import AmbiguousVersionError

db = configure_database(":memory:", ["subject", "session"])

class Raw(BaseVariable): pass
class Filtered(BaseVariable): pass
class Spikes(BaseVariable): pass

def bandpass_filter(signal, low_hz):
    return signal * low_hz  # dummy computation

def detect(signal, threshold):
    return signal * threshold  # dummy computation

print("=== Saving Raw ===")
Raw.save(np.array([1.0, 2.0, 3.0]), subject="S01", session="1")
raw_loaded = Raw.load(subject="S01", session="1")
print(f"raw.branch_params = {raw_loaded.branch_params}")

print("\n=== For each: bandpass_filter with low_hz=20 and 30 ===")
for_each(bandpass_filter, {"signal": Raw, "low_hz": 20}, [Filtered],
         subject=["S01"], session=["1"])
for_each(bandpass_filter, {"signal": Raw, "low_hz": 30}, [Filtered],
         subject=["S01"], session=["1"])

versions = db.list_versions(Filtered, subject="S01", session="1")
print(f"Filtered versions count: {len(versions)}")
assert len(versions) == 2, f"Expected 2 versions, got {len(versions)}"
for v in versions:
    print(f"  branch_params={v['branch_params']}, record_id={v['record_id'][:8]}...")

print("\n=== For each: detect with threshold=0.5 and 0.6 ===")
for_each(detect, {"signal": Filtered, "threshold": 0.5}, [Spikes],
         subject=["S01"], session=["1"])
for_each(detect, {"signal": Filtered, "threshold": 0.6}, [Spikes],
         subject=["S01"], session=["1"])

versions = db.list_versions(Spikes, subject="S01", session="1")
print(f"Spikes versions count: {len(versions)}")
assert len(versions) == 4, f"Expected 4 variants, got {len(versions)}"
for v in versions:
    print(f"  branch_params={v['branch_params']}, record_id={v['record_id'][:8]}...")

print("\n=== Load without filter → should raise AmbiguousVersionError ===")
try:
    Spikes.load(subject="S01", session="1")
    print("ERROR: should have raised AmbiguousVersionError")
    sys.exit(1)
except AmbiguousVersionError as e:
    print(f"Got expected AmbiguousVersionError: {str(e)[:100]}...")

print("\n=== Load with low_hz=20, threshold=0.5 ===")
s = Spikes.load(subject="S01", session="1", low_hz=20, threshold=0.5)
print(f"Loaded spike: branch_params={s.branch_params}")
assert s.branch_params.get("bandpass_filter.low_hz") == 20, f"Expected low_hz=20, got {s.branch_params}"
assert s.branch_params.get("detect.threshold") == 0.5, f"Expected threshold=0.5, got {s.branch_params}"

print("\n=== list_versions filtered by low_hz=20 ===")
low20 = db.list_versions(Spikes, subject="S01", session="1", low_hz=20)
print(f"Filtered by low_hz=20: {len(low20)} variants")
assert len(low20) == 2, f"Expected 2 variants for low_hz=20, got {len(low20)}"

print("\n=== list_pipeline_variants ===")
variants = db.list_pipeline_variants()
for v in variants:
    print(f"  {v['function_name']} → {v['output_type']}: constants={v['constants']}, n={v['record_count']}")

print("\n=== All assertions passed! ===")
