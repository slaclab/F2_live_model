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
from threading import Event

from epics import get_pv
from p4p.nt import NTTable, NTScalar, NTNDArray
from p4p.server import Server as PVAServer
from p4p.server.thread import SharedPV, Handler
from p4p.rpc import WorkQueue

from bmad import BmadLiveModel


PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
DIR_CONFIG = os.path.join(os.path.join(DIR_SELF, 'F2_live_model', 'config'))
sys.path.append(DIR_SELF)
os.chdir(DIR_SELF)
with open(os.path.join(DIR_CONFIG, 'facet2e.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)


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
    ("region", "s"), ("EREF","d"), ("EACT","d"), ("EERR","d"), ("BLEM_EXTANT","d"), ("BLEM_DESIGN", "d"),
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
        self.log_level = log_level
        self.log_handler = log_handler
        self.pv_root = f"{CONFIG['server']['PV']['provider_stem']}:{CONFIG['name']}"
        self.PV_heartbeat1 = get_pv(CONFIG['server']['PV']['heartbeat1'])
        self.PV_heartbeat2 = get_pv(CONFIG['server']['PV']['heartbeat2'])
        self._interrupt = Event()

    def _load_static_device_data(self):
        # store a list of static device info - this gets reused a lot
        # must be called after model initialization
        self._static_device_data = []
        for i, ele_name in enumerate(self.model.elements):
            self._static_device_data.append({
                'element': ele_name,
                'device_name': self.model.device_names[i],
                's': self.model.S[i],
                'z': self.model.Z[i],
                'length': self.model.L[i],
                })

    def _run_main_server(self):
        # run the primary server with the real-time model that publishes LEM/Twiss data
        self.model = BmadLiveModel(
            design_only=self.design_only,
            log_level=self.log_level,
            log_handler=self.log_handler,
            )
        self._load_static_device_data()
        self.model.start()

        # initialize all PVs with their design values
        design_twiss = self._get_twiss_table(which='design')
        PV_twiss_design = SharedPV(nt=NTT_TWISS, initial=design_twiss)
        PV_twiss_live =   SharedPV(nt=NTT_TWISS, initial=design_twiss)

        # special read/write array PV for saving the last LEM energy profile
        self.PV_LEM_prof = SharedPV(nt=NTNDArray(), initial=self._init_LEM_profile())

        PV_LEM_data =     SharedPV(nt=NTT_LEM_DATA, initial=self._get_LEM_table())
        PV_LEM_fudges =   [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]
        PV_LEM_ampls =    [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]
        PV_LEM_chirps =   [SharedPV(nt=NTScalar('d'), initial=1.0) for _ in range(4)]

        @self.PV_LEM_prof.put
        def onPut(pv, op):
            pv.post(op.value())
            op.done()

        self.provider = {
            f'{self.pv_root}:DESIGN:TWISS': PV_twiss_design,
            f'{self.pv_root}:LIVE:TWISS':   PV_twiss_live,
            f'{self.pv_root}:LEM:DATA':     PV_LEM_data,
            f'{self.pv_root}:LEM:L0_FUDGE': PV_LEM_fudges[0],
            f'{self.pv_root}:LEM:L1_FUDGE': PV_LEM_fudges[1],
            f'{self.pv_root}:LEM:L2_FUDGE': PV_LEM_fudges[2],
            f'{self.pv_root}:LEM:L3_FUDGE': PV_LEM_fudges[3],
            f'{self.pv_root}:LEM:L0_AMPL':  PV_LEM_ampls[0],
            f'{self.pv_root}:LEM:L1_AMPL':  PV_LEM_ampls[1],
            f'{self.pv_root}:LEM:L2_AMPL':  PV_LEM_ampls[2],
            f'{self.pv_root}:LEM:L3_AMPL':  PV_LEM_ampls[3],
            f'{self.pv_root}:LEM:L0_CHIRP': PV_LEM_chirps[0],
            f'{self.pv_root}:LEM:L1_CHIRP': PV_LEM_chirps[1],
            f'{self.pv_root}:LEM:L2_CHIRP': PV_LEM_chirps[2],
            f'{self.pv_root}:LEM:L3_CHIRP': PV_LEM_chirps[3],
            f'{self.pv_root}:LEM:PROFILE':  self.PV_LEM_prof,
            }
        with PVAServer(providers=[self.provider]):
            hb = 0
            while not self._interrupt.wait(CONFIG['server']['poll_interval']):
                hb = np.mod(hb + 1, 100)
                self.PV_heartbeat1.put(hb, 100)
                if self.design_only: continue
                PV_twiss_live.post(self._get_twiss_table(which='model'))
                PV_LEM_data.post(self._get_LEM_table())
                for i, region in enumerate(self.model.LEM):
                    sevr = 0
                    if region.fudge > 1.05:
                        sevr = 1
                    if region.fudge > 1.1:
                        sevr = 2
                    PV_LEM_ampls[i].post(region.amplitude*1e-6)
                    PV_LEM_chirps[i].post(region.chirp*1e-6)
                    PV_LEM_fudges[i].post(region.fudge, severity=sevr)
            else:
                self.model.stop()

    def _run_RMAT_server(self):
        # run the secondary server that published RMATs for the whole machine
        # setup a second instanced model just for making Rmats
        self.model = BmadLiveModel(
            instanced=True,
            log_level=self.log_level,
            log_handler=self.log_handler,
            )
        self._load_static_device_data()
        design_rmat = self._get_rmat_table(which='design', combined=True)
        design_urmat = self._get_rmat_table(which='design', combined=False)
        PV_rmat_design =  SharedPV(nt=NTT_RMAT, initial=design_rmat)
        PV_rmat_live =    SharedPV(nt=NTT_RMAT, initial=design_rmat)
        PV_urmat_design = SharedPV(nt=NTT_RMAT, initial=design_urmat)
        PV_urmat_live =   SharedPV(nt=NTT_RMAT, initial=design_urmat)
        self.provider = {
            f'{self.pv_root}:DESIGN:RMAT':  PV_rmat_design,
            f'{self.pv_root}:LIVE:RMAT':    PV_rmat_live,
            f'{self.pv_root}:DESIGN:URMAT': PV_urmat_design,
            f'{self.pv_root}:LIVE:URMAT':   PV_urmat_live,
            }
        with PVAServer(providers=[self.provider]):
            hb = 0
            while not self._interrupt.wait(CONFIG['server']['poll_interval']):
                hb = np.mod(hb + 1, 100)
                self.PV_heartbeat2.put(hb, 100)
                if self.design_only: continue
                self.model.refresh_all()
                PV_rmat_live.post(self._get_rmat_table(which='model', combined=True))
                PV_urmat_live.post(self._get_rmat_table(which='model', combined=False))

    def run(self, rmats=False):
        """
        connect the BmadLiveModel to the accelerator and begins updating PVs with live data

        :param primary: (optional) set to False to run the RMAT server instead
        :note: this function will execute forever until either an exception occurs or a 
            ``KeyboardInterrupt`` is provided to signal a stop
        """
        logging.info('Starting FACET Bmad live model service ...')
        logging.info(f' --> Running the {"primary" if not rmats else "RMAT"} server ...')
        runner = self._run_main_server if not rmats else self._run_RMAT_server
        try:
            runner()
        except KeyboardInterrupt:
            self._interrupt.set()
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
                'p0c':     model_data.p0c[i]*1e-6,
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
        # RMATs = self.model.get_all_rmats(combined=combined)
        rows = []
        RMATs = np.ndarray((len(self.model.elements),6,6))
        for i, element in enumerate(self.model.elements):
            ix_ele = (1, i) if combined else i
            R, _ = self.model.get_rmat(ix_ele, which=which)
            RMATs[i] = R

        for ix_ele, static_params in enumerate(self._static_device_data):
            # pack the 6x6 matrix into a dict with keys like 'r11', 'r12', ...
            rmat_dict = {}
            for i in range(6):
                for j in range(6):
                    rmat_dict[f'r{i+1}{j+1}'] = RMATs[ix_ele][i][j]
            rows.append({**static_params, **rmat_dict})

        return NTT_RMAT.wrap(rows)

    def _get_LEM_table(self):

        saved_fullprof = self.PV_LEM_prof.current()
        ESAVE= {
            'L0': [],
            'L1': [],
            'L2': [],
            'L3': [],
            }
        j = 0
        for region in self.model.LEM:
            for i, ele in enumerate(region.elements):
                ESAVE[region.name].append(saved_fullprof[j])
                j = j+1

        rows = []
        for region in self.model.LEM:
            for i, ele in enumerate(region.elements):
                i_global = self.model.ix[ele]
                static_params = self._static_device_data[i_global]
                elem_bdes = get_pv(f'{region.device_names[i]}:BDES').value
                E_scalar = region.EACT[i]*1e-6 / ESAVE[region.name][i]
                rows.append({
                    **static_params,
                    "region": region.name,
                    "EREF" : region.EREF[i]*1e-6,
                    "EACT" : region.EACT[i]*1e-6,
                    "EERR" : region.EERR[i]*1e-6,
                    "BLEM_EXTANT" : (region.EACT[i]*1e-6 / ESAVE[region.name][i]) * elem_bdes,
                    "BLEM_DESIGN" : region.BLEM[i],
                    })
        return NTT_LEM_DATA.wrap(rows)

    def _init_LEM_profile(self):
        prof = []
        for region in self.model.LEM:
            for i, ele in enumerate(region.elements):
                prof.append(region.EREF[i]*1e-6)
        return prof
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live model service")
    parser.add_argument(
        '--design_only', help='Serve design model data only. "LIVE" PVs will be static.',
        action='store_true'
        )
    parser.add_argument(
        '--rmats', help='Run the alternate server process that updates live R matrices',
        action='store_true'
        )
    args = parser.parse_args()

    srv_type = "primary" if args.rmats else "RMAT"
    timestamp = datetime.today().strftime(CONFIG['ts_fmt'])

    stream_handler = logging.StreamHandler()
    logfile_handler = RotatingFileHandler(
        os.path.join(CONFIG['dirs']['server_logs'], f"{CONFIG['name']}_{srv_type}_{timestamp}.log"),
        maxBytes=CONFIG['server']['logs']['max_MB']*1024*1024,
        backupCount=CONFIG['server']['logs']['N_backup'],
        )
    logging.basicConfig(
        handlers=[stream_handler, logfile_handler],
        level=CONFIG['server']['logs']['level'],
        format=CONFIG['server']['logs']['fmt'],
        datefmt=CONFIG['dt_fmt'],
        force=True
        )

    service = f2LiveModelServer(
        design_only=args.design_only,
        log_level=CONFIG['server']['logs']['level'],
        log_handler=logfile_handler,
        )
    service.run(rmats=args.rmats)

