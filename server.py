# adapted by zack from dcesar
# adapted by dcesar from https://github.com/slaclab/lcls_live_model/


import os
import sys
import logging
import argparse
import numpy as np
import time
import yaml
from datetime import datetime
from logging.handlers import RotatingFileHandler

from epics import get_pv
from p4p.nt import NTTable, NTScalar
from p4p.server import Server as PVAServer
from p4p.server.thread import SharedPV

from bmad import BmadLiveModel


PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)
os.chdir(DIR_SELF)
with open('config/facet2e.yaml') as f: CONFIG = yaml.safe_load(f)


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

NTT_LEM_DATA = NTTable([
    ("element", "s"), ("device_name", "s"), ("s", "d"), ("z", "d"), ("length", "d"),
    ("region", "s"), ("EREF","d"), ("EACT","d"), ("EERR","d"), ("BDES","d"), ("BLEM","d"),
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

    def __init__(self, design_only=False, log_level='INFO', log_handler=None):

        self.design_only = design_only

        logging.info("Initializing BmadLiveModel ...")
        self.model = BmadLiveModel(
            design_only=self.design_only,
            log_level=log_level,
            log_handler=log_handler
            )

        # store a list of static device info - this gets reused a lot
        self._static_device_data = []
        for i, ele_name in enumerate(self.model.elements):
            self._static_device_data.append({
                'element': ele_name,
                'device_name': self.model.device_names[i],
                's': self.model.S[i],
                'z': self.model.Z[i],
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

        self.PV_LEM_data =   SharedPV(nt=NTT_LEM_DATA, initial=None)
        self.PV_LEM_fudges = [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]
        self.PV_LEM_ampls =  [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]
        self.PV_LEM_chirps = [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]

        # mapping of PV names to SharedPV objects for use by the PVAServer
        pv_root = f"{CONFIG['server']['PV']['provider_stem']}:{CONFIG['name']}"
        self.provider = {
            f'{pv_root}:DESIGN:TWISS': self.PV_twiss_design,
            f'{pv_root}:LIVE:TWISS':   self.PV_twiss_live,
            f'{pv_root}:DESIGN:RMAT':  self.PV_rmat_design,
            f'{pv_root}:LIVE:RMAT':    self.PV_rmat_live,
            f'{pv_root}:DESIGN:URMAT': self.PV_urmat_design,
            f'{pv_root}:LIVE:URMAT':   self.PV_urmat_live,

            f'{pv_root}:LEM:DATA':     self.PV_LEM_data,
            f'{pv_root}:LEM:L0_FUDGE': self.PV_LEM_fudges[0],
            f'{pv_root}:LEM:L1_FUDGE': self.PV_LEM_fudges[1],
            f'{pv_root}:LEM:L2_FUDGE': self.PV_LEM_fudges[2],
            f'{pv_root}:LEM:L3_FUDGE': self.PV_LEM_fudges[3],
            f'{pv_root}:LEM:L0_AMPL':  self.PV_LEM_ampls[0]
            f'{pv_root}:LEM:L1_AMPL':  self.PV_LEM_ampls[1]
            f'{pv_root}:LEM:L2_AMPL':  self.PV_LEM_ampls[2]
            f'{pv_root}:LEM:L3_AMPL':  self.PV_LEM_ampls[3]
            f'{pv_root}:LEM:L0_CHIRP': self.PV_LEM_chirps[0],
            f'{pv_root}:LEM:L1_CHIRP': self.PV_LEM_chirps[1],
            f'{pv_root}:LEM:L2_CHIRP': self.PV_LEM_chirps[2],
            f'{pv_root}:LEM:L3_CHIRP': self.PV_LEM_chirps[3],
            }

        self.PV_heartbeat = get_pv(CONFIG['server']['PV']['heartbeat'])

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
                    time.sleep(CONFIG['server']['poll_interval'])
                    self.PV_heartbeat.put(hb, 100)
                    if self.design_only: continue
                    self.PV_twiss_live.post(self._get_twiss_table(which='model'))
                    self.PV_rmat_live.post(self._get_rmat_table(which='model', combined=True))
                    self.PV_urmat_live.post(self._get_rmat_table(which='model', combined=False))
                    for i, region in enumerate(self.model.LEM):
                        self.PV_LEM_ampls[i].post(region.amplitude)
                        self.PV_LEM_chirps[i].post(region.chirp)
                        self.PV_LEM_fudges[i].post(region.fudge)
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

    def _get_LEM_table(self):
        rows = []
        for region in self.model.LEM:
            for i, ele in enumerate(region.elements):
                i_global = self.model.ix[ele]
                rows.append({
                    **static_params[i_global]
                    "region": region.name,
                    "EREF" : region.EREF[i],
                    "EACT" : region.EACT[i],
                    "EERR" : region.EERR[i],
                    "BDES" : region.BDES[i],
                    "BLEM" : region.BLEM[i],
                    })
        return NTT_LEM_DATA.wrap(rows)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live model service")
    parser.add_argument(
        '--design_only', help='Serve design model data only. "LIVE" PVs will be static.',
        action='store_true'
        )
    args = parser.parse_args()

    timestamp = datetime.today().strftime(CONFIG['ts_fmt'])
    stream_handler = logging.StreamHandler()
    logfile_handler = RotatingFileHandler(
        os.path.join(CONFIG['dirs']['server_logs'], f"{CONFIG['name']}_model_server_{timestamp}.log"),
        maxBytes=CONFIG['server']['logs']['max_MB']*1024*1024,
        backupCount=CONFIG['server']['logs']['N_backup'],
        )
    logging.basicConfig(
        handlers=[stream_handler, logfile_handler],
        level=args.log_level,
        format=CONFIG['server']['logs']['fmt'],
        datefmt=CONFIG['dt_fmt'],
        force=True
        )

    service = f2LiveModelServer(
        design_only=args.design_only,
        log_level=CONFIG['server']['logs']['level'],
        log_handler=logfile_handler
        )

    service.run()

