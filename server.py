# adapted by zack from dcesar
# adapted by dcesar from https://github.com/slaclab/lcls_live_model/


import os
import sys
import logging
import argparse
import numpy as np
import time

from epics import get_pv
from p4p.nt import NTTable
from p4p.server import Server as PVAServer
from p4p.server.thread import SharedPV

from bmad import BmadLiveModel


PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)
os.chdir(DIR_SELF)

DIR_MODEL_DATA = '/u1/facet/physics/F2_live_model/'
DIR_SERVER_LOGS = os.path.join(DIR_MODEL_DATA, 'logs')

HEARTBEAT_CHANNEL = 'PHYS:SYS1:1:MODEL_SERVER'
MODEL_NAME = 'FACET2E'
TABLE_PV_STEM = f'BMAD:SYS0:1:{MODEL_NAME}'

SERVER_UPDATE_INTERVAL = 1.0

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
    """
    This class is used to run the live model PVA server. It uses a ``BmadLiveModel`` to
        periodically update NTTable PVs with model data.

    This service publishes PVs of the following form: ``BMAD:SYS0:1:<source>:<data>`` where
        source is ``LIVE`` or ``DESIGN`` and data is ``TWISS``, ``RMAT`` or ``URMAT``.

    :note: Running ``server.py`` as a script will start the service.

    :note: When active, this service will update a heartbeat PV: ``PHYS:SYS1:1:MODEL_SERVER``
    """

    def __init__(self, design_only=False, log_level='INFO', log_path=DIR_SERVER_LOGS):

        self.design_only = design_only

        logging.info("Initializing BmadLiveModel ...")
        self.model = BmadLiveModel(design_only=design_only, log_level=log_level)

        # store a list of static device info - this gets reused a lot
        self._static_device_data = []
        for i, ele_name in enumerate(self.model.names):
            self._static_device_data.append({
                'element': ele_name,
                'device_name': self.model.channels[i],
                's': self.model.S[i],
                'z': self.model.S[i], # s == z for now ...
                'length': self.model.L[i],
                })

        # initialize all PVs with their design values
        design_twiss = self._get_twiss_table(which='design')
        design_rmat = self._get_rmat_table(which='design', combined=True)
        design_urmat = self._get_rmat_table(which='design', combined=False)

        self.PV_twiss_design = SharedPV(nt=NTT_TWISS, initial=design_twiss)
        self.PV_twiss_live =   SharedPV(nt=NTT_TWISS, initial=design_twiss)
        self.PV_rmat_design =  SharedPV(nt=NTT_RMAT, initial=design_rmat)
        self.PV_rmat_live =    SharedPV(nt=NTT_RMAT, initial=design_rmat)
        self.PV_urmat_design = SharedPV(nt=NTT_RMAT, initial=design_urmat)
        self.PV_urmat_live =   SharedPV(nt=NTT_RMAT, initial=design_urmat)

        # mapping of PV names to SharedPV objects for use by the PVAServer
        self.provider = {
            f'{TABLE_PV_STEM}:DESIGN:TWISS': self.PV_twiss_design,
            f'{TABLE_PV_STEM}:LIVE:TWISS':   self.PV_twiss_live,
            f'{TABLE_PV_STEM}:DESIGN:RMAT':  self.PV_rmat_design,
            f'{TABLE_PV_STEM}:LIVE:RMAT':    self.PV_rmat_live,
            f'{TABLE_PV_STEM}:DESIGN:URMAT': self.PV_urmat_design,
            f'{TABLE_PV_STEM}:LIVE:URMAT':   self.PV_urmat_live,
            }

        self.PV_heartbeat = get_pv(HEARTBEAT_CHANNEL)

    def run(self):
        """
        connect the BmadLiveModel to the accelerator and begins updating PVs with live data

        :note: this function will execute forever until either an exception occurs or a 
            ``KeyboardInterrupt`` is provided to signal a stop
        """
        logging.info('Starting FACET Bmad live model service ...')
        with self.model, PVAServer(providers=[self.provider]):
            try:
                hb = 0
                while True:
                    hb = np.mod(hb + 1, 100)
                    time.sleep(SERVER_UPDATE_INTERVAL)
                    self.PV_heartbeat.put(hb, 100)
                    if self.design_only: continue
                    self.PV_twiss_live.post(self._get_twiss_table(which='model'))
                    self.PV_rmat_live.post(self._get_rmat_table(which='model', combined=True))
                    self.PV_urmat_live.post(self._get_rmat_table(which='model', combined=False))
            except KeyboardInterrupt:
                pass
            finally:
                logging.info("Stopping service.")

    def _get_twiss_table(self, which='model'):
        # returns a table of twiss parameters at each element
        if which == 'model':
            model_data = self.model.live
        else:
            model_data = self.model.design
        rows = []
        for i, static_params in enumerate(self._static_device_data):
            rows.append({
                **static_params,
                'p0c':     model_data.p0c[i],
                'alpha_x': model_data.twiss.alpha_x[i],
                'beta_x':  model_data.twiss.beta_x[i],
                'eta_x':   model_data.twiss.eta_x[i],
                'etap_x':  model_data.twiss.etap_x[i],
                'psi_x':   model_data.twiss.psi_x[i],
                'alpha_y': model_data.twiss.alpha_y[i],
                'beta_y':  model_data.twiss.beta_y[i],
                'eta_y':   model_data.twiss.eta_y[i],
                'etap_y':  model_data.twiss.etap_y[i],
                'psi_y':   model_data.twiss.psi_y[i],
                })
        return NTT_TWISS.wrap(rows)

    def _get_rmat_table(self, which='model', combined=False):
        # makes a table of rmats for all elements -- single-element maps by default,
        # if the 'combined' flag is set, will calculate the maps from the first element,
        rows = []
        for ix_ele, static_params in enumerate(self._static_device_data):
            elem = (1, ix_ele) if combined else ix_ele
            R, _ = self.model.get_rmat(elem, which=which)

            # pack the 6x6 matrix into a dict with keys like 'r11', 'r12', ...
            rmat_dict = {}
            for i in range(6):
                for j in range(6):
                    rmat_dict[f'r{i+1}{j+1}'] = R[i][j]

            rows.append({**static_params, **rmat_dict})

        return NTT_RMAT.wrap(rows)

    
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

    log_file_path = os.path.join(args.log_dir, 'live_model.log')

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    logging.basicConfig(
        handlers=[stream_handler, file_handler],
        level=args.log_level,
        format="%(asctime)s.%(msecs)03d [F2ModelServer] %(levelname)s: %(message)s ",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
        )

    service = f2LiveModelServer(
        design_only=args.design_only,
        log_level=args.log_level,
        log_path=log_file_path
        )

    service.run()

