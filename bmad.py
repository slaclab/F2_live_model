# Bmad live model object
# for use by programmers and the model server

# TODO:
# - add solenoid, sextupole and other device monitors
# - figure out dipole unit conversions


import os
import sys
import time
import logging
import numpy as np
from copy import deepcopy
from traceback import print_exc
from threading import Lock, Thread, Event, get_native_id
from functools import cache, partial
from queue import SimpleQueue, Empty

from epics import get_pv
from pytao import Tao

PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)

from structs import _ModelData, _Twiss, _Cavity, _Quad, _Corrector, _Dipole

DIR_F2_LATTICE = '/usr/local/facet/tools/facet2-lattice/'
TAO_INIT_F2_DESIGN = os.path.join(DIR_F2_LATTICE, 'bmad/models/f2_elec/tao.init')
os.environ['FACET2_LATTICE'] = DIR_F2_LATTICE

# waiting period in between calls to the update_routine in _background_loop
MODEL_POLL_INTERVAL = 0.05

# string patterns for linac cavities
CAV_STR_L1 = ["K11_1*", "K11_2*"]
CAV_STR_L2 = ["K11_4*", "K11_5*", "K11_6*", "K11_7*", "K11_8*", "K12_*", "K13_*", "K14_*"]
CAV_STR_L3 = ["K15_*", "K16_*", "K17_*", "K18_*", "K19_*"]

BAD_KLYS = ['K11_1', 'K11_2', 'K11_3', 'K13_2', 'K14_7', 'K15_2', 'K19_7', 'K19_8']


# TODO: find a better home for this caculation
def intkGm_2_gradTm(B_int, l): return -1 * 0.1 * B_int / l


class BmadLiveModel:
    """
    Provides an instance of PyTao that is updated with live machine parameters.
    This class streams live accelerator settings data via asynchronus network monitoring
    and a daemon process that periodically updates Tao.
    
    Has a limited API for common tasks, more sophisitcated tasks can manipulate the tao instance

    :param design_only: disables connection to the controls system, defaults to False
    :param instanced: take single-shot live data instead of streaming, defaults to False
    :param log_level: desired logging level, defaults to 'INFO'
    :param FileHandler: (optional) FileHandler object for logging, otherwise logs to stdout

    :raises ValueError: if ``design_only`` and ``instanced`` flags are both set
    """

    def __init__(self, design_only=False, instanced=False, log_level='INFO', FileHandler=None):
        
        if design_only and instanced:
            raise ValueError('"design_only" and "instanced" models are mutually exclusive.')

        self.log_level = log_level
        self._design_only, self._instanced = design_only, instanced
        self._streaming = (not self._design_only) and (not self._instanced)

        # TODO: setup rotatingFileHandler here if arg is supplied ...
        self.log = logging.getLogger(__name__)
        logging.basicConfig(
            stream=sys.stdout, level=self.log_level,
            format="%(asctime)s.%(msecs)03d [BmadLiveModel] %(levelname)s: %(message)s ",
            datefmt="%Y-%m-%d %H:%M:%S", force=True
            )

        self.log.info(f'Building FACET2E model ...')
        self.tao = Tao(f'-init {TAO_INIT_F2_DESIGN} -noplot')
       
        # initialize self.design & self.live using design model data
        self._init_static_lattice_info()

        p0c_design = self._lat_list_array('ele.p0c')
        E0_design = self._lat_list_array('ele.e_tot')
        tw_design = self._fetch_twiss(which='design')
        self._design_model_data = _ModelData(p0c=p0c_design, e_tot=E0_design, twiss=tw_design)

        # TODO: add dipoles, other stuff to simulation
        self._init_rf()
        self._init_quads()
        # self._init_bends()
        # self._init_cors()
        # self._init_misc()

        if design_only:
            self._live_model_data = None
            self.log.warning('Serving static design model only.')
            return

        self.log.info('Initialized static model data.')

        self._live_model_data = deepcopy(self._design_model_data)
        self._mutex = Lock()
        self._tao_cmd_queue = SimpleQueue()
        if self._instanced: self._init_machine_connection()
        
    def __enter__(self):
        self.start()
        return self

    def start(self):
        """
        starts daemon to monitor accelerator controls data & update PyTao

        :raises RuntimeError: if the ``design_only`` or ``instanced`` flags are set
        """
        if not self._streaming:
            raise RuntimeError('Live data unavailable for instanced/design models')
        self._init_machine_connection()
        self.log.info('Starting model-update daemon ...')
        self._interrupt = Event()
        self._model_daemon = Thread(daemon=True, target=self._background_model_update)
        self._model_daemon.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        kb_interrupt = exc_type is KeyboardInterrupt
        exit_OK = (exc_type is None) or kb_interrupt
        if kb_interrupt: self.log.info('KeyboardInterrupt detected')
        elif not exit_OK: self.log.critical(f'FATAL ERROR: {exc_type} {exc_val}')
        self.stop()
        return exit_OK

    def stop(self):
        """
        disconnects from all PVs and stops the background update thread
        """
        self.log.info('Stopping model-update daemon ...')
        self._interrupt.set()
        self._model_daemon.join()

    def refresh_all(self, catch_errs=False):
        """
        single-shot model update (only for use with instanced models)

        :param catch_errs: catch errors and log during update rather than halt, defaults to False

        :raises RuntimeError: if the ``design_only`` flag is set, or the ``instanced`` flag is not set
        """
        if self._streaming: raise RuntimeError('refresh_all only usable in instanced mode')
        self._refresh(catch_errs=catch_errs)

    def write_bmad(self, title=None):
        """
        save current lattice to a .bmad file, default title is ``f2_elec_<ymdhms>.bmad``

        :param title: absolute filepath for desired output file, default is the current directory
        """
        if not title: title = f'f2_elec_{datetime.today().strftime("%Y%m%d%H%M%S")}.bmad'
        self.tao.cmd(f'write bmad -format one_file {title}')
        self.log.info(f'Lattice data written to {title}')

    def _init_machine_connection(self):
        try:
            self.log.info('Connecting to accelerator controls system ...')
            self._refresh(catch_errs=False, attach_callbacks=self._streaming)
        except Exception as err:
            self.log.critical('FATAL ERROR during live model initialization')
            raise err

    def _refresh(self, catch_errs=False, attach_callbacks=False):
        # explicitly updates ALL machine parameters, or attaches callbacks to PVs for streaming
        tasks = [
            Thread(target=partial(self._fetch_rf, attach_callbacks=attach_callbacks)),
            Thread(target=partial(self._fetch_quads, attach_callbacks=attach_callbacks)),
            Thread(target=partial(self._fetch_cors, attach_callbacks=attach_callbacks)),
            Thread(target=partial(self._fetch_misc, attach_callbacks=attach_callbacks)),
            ]
        try:
            t_st = time.time()
            for th in tasks: th.start()
            for th in tasks: th.join()
            for th in tasks: assert not th.is_alive()
            self._update_model()
            t_el = time.time() - t_st
            self.log.info(f'Model refreshed. Time elapsed: {t_el:.6f}s')
        except Exception as err:
            msg = f'data refresh failed ({repr(err)})'
            if catch_errs: self.log.warnings(msg)
            else:
                self.log.critical(msg)
                raise err

    def _update_model(self):
        id_str = f'[model-update@{get_native_id()}]'
        
        # check how many commands are in the queue, short circuit if it's empty
        N_update = self._tao_cmd_queue.qsize()
        if not N_update: return

        # unload the queue
        # device_updates tracks what is changed to update _ModelData attributes in-kind
        t_st = time.time()
        update_cmds, device_updates = [], []
        try:
            for _ in range(N_update):
                cmd = self._tao_cmd_queue.get_nowait()
                cmd_sp = cmd.split()
                name = cmd_sp[2]
                attr = cmd_sp[3]
                value = float(cmd_sp[-1])
                device_updates.append((name,attr,value))
                update_cmds.append(cmd)
        except Empty:
            pass

        # tao.cmds automatically disables lattice recalculation until all updates are submitted
        # -> no need to call set global lattice_calc_on before/after
        self.tao.cmds(update_cmds)

        # update arrays & device dictionaries
        with self._mutex:
            self._live_model_data.p0c   = self._lat_list_array('ele.p0c')
            self._live_model_data.e_tot = self._lat_list_array('ele.e_tot')
            self._live_model_data.twiss = self._fetch_twiss()
            for (name,attr,value) in device_updates:
                # determine which family of device was changed based on the attribute type
                # easier than checking & comparing names
                if attr in ['voltage','phi0']: dev = self._live_model_data.rf[name]
                elif attr == 'b_field':        dev = self._live_model_data.bends[name]
                elif attr == 'kick':           dev = self._live_model_data.cors[name]
                elif attr == 'b1_gradient':    dev = self._live_model_data.quads[name]
                setattr(dev, attr, value)

        t_el = time.time() - t_st
        self.log.info(f'{id_str} Updated {N_update} model parameters in {t_el:.4f}s')
        return

    def _background_model_update(self):
        """ daemon task, runs self._update_model until self. _interrupt is set """
        id_str = f'[model-update@{get_native_id()}]'
        while not self._interrupt.wait(MODEL_POLL_INTERVAL):
            try:
                self._update_model()
            except Exception as err:
                self.log.info(f'{id_str} iteration FAILED: {repr(err)}')
        else:
            self.log.info(f'{id_str} Received interrupt signal, updating stopped.')

    # serial loops to get device settings, called on initialization and by _refresh
    # for streaming data, these functions will attach _submit_update functions as per device
    # callbacks rather than simply calling the update functions directly
    # TODO: these functions are all very similar, possible to encapsulate better?

    def _fetch_rf(self, attach_callbacks=False):

        # TODO: L0-A setup (?)

        # TODO: L0-B setup

        # TODO: L1 setup

        # L2/L3 klystrons are repetetive
        for kname, rfs in self.klys_structure_map.items():
            if kname in BAD_KLYS: continue
            k_ch = self._klys_channels[kname]
            self.log.debug(f'Monitoring {kname:10s} via C/A address: {k_ch}')

            pv_enld = get_pv(f'{k_ch}:ENLD')
            pv_pdes = get_pv(f'{k_ch}:PDES')

            if attach_callbacks:
                pv_enld.clear_callbacks()
                pv_pdes.clear_callbacks()
                pv_enld.add_callback(partial(self._submit_update_rf_ampl, ele=kname))
                pv_pdes.add_callback(partial(self._submit_update_rf_phase, ele=kname))
            else:
                self._submit_update_rf_ampl(value=pv_enld.value, ele=kname)
                self._submit_update_rf_phase(value=pv_pdes.value, ele=kname)

    def _fetch_quads(self, attach_callbacks=False):
        for qname in self.names[self._ix['QUAD']]:
            i_q = self._ix[qname]
            q_ch = self._channels[i_q]
            if q_ch == '': continue
            self.log.debug(f'Monitoring {qname:10s} via C/A address: {q_ch}')

            pv_bdes = get_pv(f'{q_ch}:BDES')

            if attach_callbacks:
                pv_bdes.clear_callbacks()
                pv_bdes.add_callback(partial(self._submit_update_quad, ele=qname))
            else:
                self._submit_update_quad(pv_bdes.value, ele=qname)

    def _fetch_cors(self, attach_callbacks=False):
        return

    def _fetch_misc(self, attach_callbacks=False):
        # update bends, solenoids, sextupoles, TCAVs?...
        for bname in self.names[self._ix['BEND']]:
            i_b = self._ix[bname]
            b_ch = self._channels[i_b]
            if b_ch == '': continue
            self.log.debug(f'Monitoring {bname:10s} via C/A address: {b_ch}')

            pv_bdes = get_pv(f'{b_ch}:BDES')

            if attach_callbacks:
                pv_bdes.clear_callbacks()
                pv_bdes.add_callback(partial(self._submit_update_bend, ele=bname))
            else:
                self._submit_update_bend(pv_bdes.value, ele=bname)
        return

    # device value update functions
    # each converts units from EPICS->Bmad as needed & submits 'set ele' command(s) to the queue

    def _submit_update_rf_enable(self, value, ele, **kw):
        return

    def _submit_update_rf_phase(self, value, ele, **kw):
        p_rfs = value / 360.
        rfs = self.klys_structure_map[ele]
        # TEMPORARY L2 kludge until SBST PVs are used
        if rfs[0] in self._cav_l2: p_rfs = p_rfs - (35./360.)
        for cav in rfs:
            self._tao_cmd_queue.put(f'set ele {cav} phi0 = {p_rfs:.9f}')

    def _submit_update_rf_ampl(self, value, ele, **kw):
        rfs = self.klys_structure_map[ele]
        # (naively) assume power is evenly-distributed across each DLWG
        V_rfs = 1e6 * (value / len(rfs))
        for cav in rfs:
            self._tao_cmd_queue.put(f'set ele {cav} voltage = {V_rfs:.4f}')

    def _submit_update_solenoid(self, value, ele, **kw):
        return

    def _submit_update_bend(self, value, ele, **kw):
        # TODO: need to convert bend units from GeV/c to Tm
        return

    def _submit_update_corrector(self, value, ele, **kw):
        # TODO: convert Bfield to kick angle (check bend radius formulae)
        return

    def _submit_update_quad(self, value, ele, **kw):
        # convert incoming integral B-field in kGm to gradient in Tm
        grad = intkGm_2_gradTm(value, self.L[self._ix[ele]])
        self._tao_cmd_queue.put(f'set ele {ele} b1_gradient = {grad:.4f}')

    def _submit_update_sextupole(self, value, ele, **kw):
        return

    def _submit_update_offset(self, value, ele, plane, **kw):
        return


    # model data are held as object properties
    # static properties (index map, names, channels, S, device lengths) are cached
    
    @property
    @cache
    def ix(self):
        """
        dictionary of numerical indicies of various beamline elements

        ``BmadLiveModel.ix['<ele_name>']`` returns numerical indices for the given element
        in model data arrays
        ex: ``BmadLiveModel.L[ix['QE10525']]`` would return the length of QE10525

        There are also some shortcut masks for quickly selecting all elements of a given type
        ``ix['QUAD']`` will return the indicies of every quadrupole magnet in the model,
        valid masks are: ``RF, SOLN, XCOR, YCOR, COR, BEND, QUAD, SEXT, DRIFT, BPMS, PROF``

        :note: mask indicies are equivalent to: ``np.where(self.ele_types == '<Bmad ele.key>')``
        """
        return self._ix

    @property
    @cache
    def names(self):
        """ Bmad model names of all elements in s-order """
        return self._lat_list_array('ele.name', dtype=str)

    @property
    @cache
    def ele_types(self):
        """ Bmad 'key' (element type) of all elements in s-order """
        return self._lat_list_array('ele.key', dtype=str)

    @property
    @cache
    def S(self):
        """ S position of all elements in s-order """
        return self._lat_list_array('ele.s')

    @property
    @cache
    def L(self):
        """ length of all elements in s-order """
        return self._lat_list_array('ele.l')

    @property
    @cache
    def channels(self):
        """ control system channel access addresse of all elements in s-order """
        return self._channels
        
    @property
    @cache
    def design(self):
        """ design model data, identical interface to live model data """
        return self._design_model_data

    @property
    def live(self):
        """
        Data structure containing live model data.

        Live momentum profile and twiss parameters are stored an Numpy arrays in s-order,
        while single-device information is accessed through a dictionary of device data structures.

        top-level attributes are:
        ``live.p0c, e_tot, gamma_rel, Brho, twiss, rf, quads, bends, cors``

        the twiss data structure contains the following fields (for x and y):
        ``twiss.beta_x, alpha_x, eta_x, etap_x, psi_x, gamma_x, ...``

        each device dictionary is indexed by element name (i.e. 'QE10525') and returns dataclasses
        describing the relevant live parameters, as well as s positions and lengths for convenience
        unique attributes are as follows:
        ``rf[<name>].voltage``, ``rf[<name>].phi0``,  ``quads[<name>].b1_gradient``

        :note: this interface is nonexhaustive, and only covers commonly used data
        """
        return self._live_model_data

    def get_rmat(self, ele, which='model'):
        """
        returns 6x6 ndarray of single-element or (if given 2 elements) A-to-B transfer maps

        :param ele: beamline element(s), may be a single element e or a tuple of (e1, e2)
        :param which: which lattice to read from, default is 'model', can also choose 'design'

        :return: (R,v0) tuple of the map R (6x6 np.ndarray), and "0th order" map v0 (1x6 vector)
        """
        if type(ele) in [tuple, list]:
            e1, e2 = ele[0], ele[1]
        else:
            e1, e2 = ele, None

        v0 = None

        if not e2 or e1 == e2:
            r = self.tao.ele_mat6(e1)
            v0 = self.tao.ele_mat6(e1, who='vec0')
            R = np.ndarray((6,6))
            for i,l in enumerate(r): R[i] = r[l]
        
        else:
            # need get map from/to downstream-most slave elements if e1/e2 are lords
            # Bmad defines taylor maps between downstream faces of elements
            if e1 in self._lords: e1 = self._slaves[e1][-1]
            if e2 in self._lords: e2 = self._slaves[e2][-1]
            r = self.tao.matrix(f'{e1}|{which}', e2)
            R = r['mat6']
            v0 = r['vec0']

        return R, v0

    def _fetch_twiss(self, elems='*', which='model'):
        _req_array = partial(self._lat_list_array, elems=elems, which=which)
        return _Twiss(
            beta_x =  _req_array('ele.a.beta'),
            beta_y =  _req_array('ele.b.beta'),
            alpha_x = _req_array('ele.a.alpha'),
            alpha_y = _req_array('ele.b.alpha'),
            eta_x =   _req_array('ele.a.eta'),
            eta_y =   _req_array('ele.b.eta'),
            etap_x =  _req_array('ele.a.etap'),
            etap_y =  _req_array('ele.b.etap'),
            psi_x =   _req_array('ele.a.phi'),
            psi_y =   _req_array('ele.b.phi'),
            )
    
    def _init_static_lattice_info(self):
        """ loads in static params: element names, positions and lord/slave config  """

        # list of all cavities in each linac
        self._cav_l0 = np.array(['L0AF', 'L0BF'])
        self._cav_l1 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CAV_STR_L1]
            )
        self._cav_l2 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CAV_STR_L2]
            )
        self._cav_l3 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CAV_STR_L3]
            )
        self._all_cavs = np.concatenate([self._cav_l0, self._cav_l1, self._cav_l2, self._cav_l3])

        # map of L1 - L3 klystron names and their associated cavities
        # Bmad only deals in individual cavities, so we need this metadata
        # L0A/B are single structures, and so not included
        self._linac_klys = []
        for cav_name in np.concatenate([self._cav_l1, self._cav_l2, self._cav_l3]):
            k_id = cav_name[:5]
            if k_id not in self._linac_klys: self._linac_klys.append(k_id)
        self.klys_structure_map = {}
        for k_id in self._linac_klys:
            self.klys_structure_map[k_id] = self._lat_list_array('ele.name', elems=f'{k_id}*')

        # the self.ix dictionary provides an interface for array indexing
        # contains indices of single elements, and index masks for element types
        self._ix = {}
        for i, e in enumerate(self.names): self._ix[e] = i

        self._ix['RF']   = np.where(self.ele_types == 'Lcavity')
        self._ix['SOLN'] = np.where(self.ele_types == 'Solenoid')
        self._ix['XCOR'] = np.where((self.ele_types == 'HKicker'))
        self._ix['YCOR'] = np.where((self.ele_types == 'VKicker'))
        self._ix['COR']  = np.sort(np.append(self._ix['XCOR'], self._ix['YCOR']))
        self._ix['BEND'] = np.where(self.ele_types == 'SBend')
        self._ix['QUAD'] = np.where(self.ele_types == 'Quadrupole')
        self._ix['SEXT'] = np.where(self.ele_types == 'Sextupole')
        self._ix['DRIFT'] = np.where(self.ele_types == 'Drift')

        # BPMs, and intercepting screens both use the key 'Monitor' in Bmad, need to filter
        # BPM names go like BPM<sec><unit> in the linac, or M<label>in the IP
        self._ix['BPMS'], self._ix['PROF'] = [], []
        for ix, e in enumerate(self.names):
            if self.ele_types[ix] != 'Monitor': continue
            if e[:3] == 'BPM' or e[0] == 'M': self._ix['BPMS'].append(ix)
            else: self._ix['PROF'].append(ix)

        # get control system names (stored as the element "alias" in Bmad)
        self._channels = np.empty(self.names.size, dtype='<U20')
        self._klys_channels = {}
        for line in self.tao.show('lattice -attr alias -all -no_slaves -no_label_lines -python'):
            ele_name = line.split(';')[1]
            ch_name = line.split(';')[-1]
            if ch_name == '': continue
            idx = self.ix[ele_name]
            self._channels[idx] = ch_name
            # also add device names to index lookup, for user convenience
            self._ix[ch_name] = idx
            # klystron channels
            if ele_name in self._all_cavs:
                k_id = ele_name[:5]
                if k_id[:3] == 'K11':
                    nk = k_id[-1]
                    ch_name = f'LI11:KLYS:{nk}1'
                if k_id not in self._klys_channels.keys():
                    self._klys_channels[k_id] = ch_name

        # some quads, cavities and other elements are split into multiple parts
        # keep a dictionary of the slave elements for use later
        # also disabling debug messages here -- too much spam for even 'debug' mode
        self.log.debug('Tagging slave elements ...')
        logging.disable(logging.DEBUG)
        self._slave_rf, self._slave_quads, self._slave_misc = {}, {}, {}
        for ele, etype in zip(self.names, self.ele_types):
            try: r = self.tao.ele_lord_slave(ele)
            except RuntimeError: continue # multi-instance elements cause errors, skip 'em
            if len(r) == 1: continue
            if etype == 'Lcavity':      slave_dict = self._slave_rf
            elif etype == 'Quadrupole': slave_dict = self._slave_quads
            else:                       slave_dict = self._slave_misc
            slave_dict[ele] = []
            for l in r:
                if l['type'] == 'Slave': slave_dict[ele].append(l['name'])
        self._lord_rf = self._slave_rf.keys()
        self._lord_quads = self._slave_quads.keys()
        self._lord_misc = self._slave_misc.keys()
        self._slaves = {**self._slave_rf, **self._slave_quads, **self._slave_misc}
        self._lords = list(self._lord_rf) + list(self._lord_quads) + list(self._lord_misc)
        logging.disable(logging.NOTSET)

    # functions to populate dictionaries of data structures with device settings
    # design setting dicts are duplicated to initialize the live _ModelData object

    def _init_rf(self):
        _req_cav = partial(self._lat_list_array, elems='lcavity::*')
        for n,S,l,V,phi in zip(
            _req_cav('ele.name'),
            _req_cav('ele.s'),
            _req_cav('ele.l'),
            _req_cav('ele.voltage'),
            _req_cav('ele.phi0'),
            ):
            # only including the forward RF for now. Also L1XF doesn't exist...
            if n in ['TCY10490', 'L1XF', 'TCY15280', 'XTCAVF']: continue
            self._design_model_data.rf[n] = _Cavity(S=S, l=l, voltage=V, phase=360.*phi)

    def _init_bends(self):
        _req_bend = partial(self._lat_list_array, elems='sbend::*')
        for n,s,l,g in zip(
            _req_bend('ele.name'),
            _req_bend('ele.s'),
            _req_bend('ele.l'),
            _req_bend('ele.b_field'),
            ):
            self._design_model_data.bends[n] = _Dipole(S=s, l=l, b_field=g)

    def _init_cors(self):
        _req_xcor = partial(self._lat_list_array, elems='hkicker::*')
        for n,s,l,k in zip(
            _req_xcor('ele.name'),
            _req_xcor('ele.s'),
            _req_xcor('ele.l'),
            _req_xcor('ele.kick'),
            ):
            self._design_model_data.cors[n] = _Corrector(S=s, l=l, kick=k)

        _req_ycor = partial(self._lat_list_array, elems='vkicker::*')
        for n,s,l,k in zip(
            _req_ycor('ele.name'),
            _req_ycor('ele.s'),
            _req_ycor('ele.l'),
            _req_ycor('ele.kick'),
            ):
            self._design_model_data.cors[n] = _Corrector(S=s, l=l, kick=k)

    def _init_quads(self):
        _req_quad = partial(self._lat_list_array, elems='quad::*')
        for n,s,l,g in zip(
            _req_quad('ele.name'),
            _req_quad('ele.s'),
            _req_quad('ele.l'),
            _req_quad('ele.b1_gradient'),
            ):
            self._design_model_data.quads[n] = _Quad(S=s, l=l, b1_gradient=g)

    def _init_misc(self):
        # sextupoles + mover offsets
        # SOL121
        return

    def _lat_list_array(self, who, elems='*', which='model', dtype=np.float64):
        # packs a single-column of lattice data from tao.lat_list as an array
        if dtype is np.float64:
            return np.array(
                self.tao.lat_list(elems, who, which=which, flags='-array_out -no_slaves')
                )
        return np.array(
            self.tao.lat_list(elems, who, which=which, flags='-no_slaves')
            ).astype(dtype)
