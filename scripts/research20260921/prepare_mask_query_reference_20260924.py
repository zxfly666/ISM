"""New held-out MC using the tested generator, without altering old source."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from ism_diffusion import geometry_study as gs
import prepare_sparse_conditioning_data as implementation
from mask_query_factorial_design import MC_SEED

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out)
    implementation.MC_SEED=MC_SEED
    try:implementation.reference(out)
    except Exception as error:
        gs.atomic_json(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise
