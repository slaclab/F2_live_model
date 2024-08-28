# adapted by zack from dcesar
# adapted by dcesar from https://github.com/slaclab/lcls_live_model/blob/master/live_model.py#L343


import os
import sys
import logging
import argparse
import threading
from typing import Optional, List, Dict, Any
from functools import lru_cache
import numpy as np
import time

import epics
from p4p.nt import NTTable
from p4p.server import Server as PVAServer
from p4p.server.thread import SharedPV
from pytao import Tao
from bmad_live import BmadLiveModel

PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(SELF_PATH)[:-1])
sys.path.append(DIR_SELF)
os.chdir(DIR_SELF)

DIR_MODEL_DATA = '/u1/facet/physics/F2_live_model/'
DIR_SERVER_LOGS = os.path.join(DIR_MODEL_DATA, 'logs')

HEARTBEAT_CHANNEL = 'PHYS:SYS1:1:MODEL_SERVER'

SERVER_UPDATE_INTERVAL = 0.5

NTT_TWISS = NTTable([
    ("element", "s"), ("device_name", "s"),
    ("s", "d"), ("z", "d"), ("length", "d"), ("p0c", "d"),
    ("alpha_x", "d"), ("beta_x", "d"), ("eta_x", "d"), ("etap_x", "d"), ("psi_x", "d"),
    ("alpha_y", "d"), ("beta_y", "d"), ("eta_y", "d"), ("etap_y", "d"), ("psi_y", "d"),
    ])

NTT_RMAT = NTTable([
    ("element", "s"), ("device_name", "s"), ("s", "d"), ("z", "d"), ("length", "d"),
    ("r11", "d"), ("r12", "d"), ("r13", "d"), ("r14", "d"), ("r15", "d"), ("r16", "d"),
    ("r21", "d"), ("r22", "d"), ("r23", "d"), ("r24", "d"), ("r25", "d"), ("r26", "d"),
    ("r31", "d"), ("r32", "d"), ("r33", "d"), ("r34", "d"), ("r35", "d"), ("r36", "d"),
    ("r41", "d"), ("r42", "d"), ("r43", "d"), ("r44", "d"), ("r45", "d"), ("r46", "d"),
    ("r51", "d"), ("r52", "d"), ("r53", "d"), ("r54", "d"), ("r55", "d"), ("r56", "d"),
    ("r61", "d"), ("r62", "d"), ("r63", "d"), ("r64", "d"), ("r65", "d"), ("r66", "d"),
    ])


class f2LiveModelServer:
    """ live model! """

    def __init__(self, design_only=False, log_level='INFO', log_path=DIR_SERVER_LOGS):

        # set up the live model
        self.model = BmadLiveModel()

        self.model_name = 'FACET2E'
        self.PV_heartbeat = get_pv(HEARTBEAT_CHANNEL)

        # setup PVs & map them to their names
        pv_twiss_design = SharedPV(nt=NTT_TWISS, initial=initial_twiss_table)
        pv_twiss_live = SharedPV(nt=NTT_TWISS, initial=initial_twiss_table)
        pv_rmat_design = SharedPV(nt=NTT_RMAT, initial=initial_rmat_table)
        pv_rmat_live = SharedPV(nt=NTT_RMAT, initial=initial_rmat_table)
        # pv_urmat_design = SharedPV(nt=NTT_RMAT, initial=initial_rmat_table) # needed?
        # pv_urmat_live = SharedPV(nt=NTT_RMAT, initial=initial_rmat_table)

        # Map the PVs to PV names
        self.provider = {
            f"BMAD:SYS0:1:{self.model_name.upper()}:LIVE:TWISS": live_twiss_pv,
            f"BMAD:SYS0:1:{self.model_name.upper()}:DESIGN:TWISS": design_twiss_pv,
            f"BMAD:SYS0:1:{self.model_name.upper()}:LIVE:RMAT": live_rmat_pv,
            f"BMAD:SYS0:1:{self.model_name.upper()}:DESIGN:RMAT": design_rmat_pv,
            f"BMAD:SYS0:1:{self.model_name.upper()}:LIVE:URMAT": live_u_rmat_pv,
            f"BMAD:SYS0:1:{self.model_name.upper()}:DESIGN:URMAT": design_u_rmat_pv
        }

    def get_twiss_table(self, which='model'):
        return

    def get_rmat_table(self, which='model', combined=False):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live model service")
    parser.add_argument(
        '--log_level', help='Configure level of log display',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
        )
    parser.add_argument(
        '--log_dir', help='Directory where logs get saved.',
        default=DIR_SERVER_LOGS
        )
    parser.add_argument(
        '--design_only', help='Serve design model data only. "LIVE" PVs will be static.',
        action='store_true'
        )
    args = parser.parse_args()

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile)
    logging.basicConfig(
        handlers=[stream_handler, file_handler],
        level=args.log_level,
        format="%(asctime)s,%(msecs)d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )

    logging.info("Starting FACET Bmad live model service ...")

    model_server = f2LiveModelServer(
        design_only=args.design_only,
        log_level=args.log_level,
        log_path=os.path.join(args.log_dir, 'live_model.log')
        )

    with PVAServer(providers=[pv_provider]), model_server:
        try:
            ii = 0
            while True:
                ii = np.mod(ii + 1, 100)
                model_server.PV_heartbeat.put(ii, 100)
                time.sleep(SERVER_UPDATE_INTERVAL)

                # update tables here!

        except KeyboardInterrupt:
            pass
        finally:
            logging.info("Stopping service.")