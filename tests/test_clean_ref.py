import os
import sys

# Ensure the project root is on sys.path for direct module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import afecd_csv_embedding as m

def test_clean_ref_trims_trailing_periods():
    assert m.clean_ref('2.1.') == '2.1'
    assert m.clean_ref('2.') == '2'
    assert m.clean_ref('2.1') == '2.1'
    assert m.clean_ref('2..1.') == '2.1'
