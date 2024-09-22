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
import yaml
from copy import deepcopy
from traceback import print_exc
from threading import Thread, Event, get_native_id
from functools import cache, partial
from queue import SimpleQueue

from epics import get_pv
from pytao import Tao

PATH_SELF = os.path.dirname(os.path.abspath(__file__))
DIR_SELF = os.path.join(*os.path.split(PATH_SELF)[:-1])
sys.path.append(DIR_SELF)
with open('config/facet2e.yaml') as f: CONFIG = yaml.safe_load(f)
os.environ['FACET2_LATTICE'] = CONFIG['dirs']['lattice']

from structs import _ModelData, _F2LEMData, _LEMRegionData, _Twiss, _Cavity, _Quad, _Dipole
from F2_pytools import slc_utils as slc


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
    def __init__(self, design_only=False, instanced=False, log_level='INFO', log_handler=None):
        if design_only and instanced:
            raise ValueError('"design_only" and "instanced" models are mutually exclusive.')

        self.log_level = log_level
        self._design_only, self._instanced = design_only, instanced
        self._streaming = (not self._design_only) and (not self._instanced)

        handlers = [logging.StreamHandler()]
        if log_handler: handlers.append(log_handler)
        self.log = logging.getLogger(__name__)
        logging.basicConfig(
            handlers=handlers, level=self.log_level,
            format=CONFIG['bmad']['logs']['fmt'],
            datefmt=CONFIG['dt_fmt'], force=True
            )

        self.log.info(f"Building {CONFIG['name']} model with Tao ...")
        self._tao = Tao(f"-init {CONFIG['bmad']['tao_init_path']} -noplot")

        # initialize static data & other structs
        self.log.info('Building data structures ...')
        self._design_model_data = _ModelData(
            p0c=self._lat_list_array('ele.p0c'),
            e_tot=self._lat_list_array('ele.e_tot'),
            twiss=self._fetch_twiss(which='design'),
            )
        self._init_static_lattice_data()
        self._init_static_LEM_data()
        self._init_static_device_data()

        if design_only:
            self._live_model_data = None
            self.log.warning('Serving static design model only.')
            return

        self._live_model_data = deepcopy(self._design_model_data)
        self._model_update_queue = SimpleQueue()
        self._lem_update_queue = SimpleQueue()
        self.log.info('Finished static initialization.')
        
        if self._instanced: self.refresh_all()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        kb_interrupt = exc_type is KeyboardInterrupt
        exit_OK = (exc_type is None) or kb_interrupt
        if kb_interrupt: self.log.info('KeyboardInterrupt detected')
        elif not exit_OK: self.log.critical(f'FATAL ERROR: {exc_type} {exc_val}')
        self.stop()
        return exit_OK

    def start(self):
        """
        starts daemon threads to monitor accelerator controls data & update PyTao

        :raises RuntimeError: if the ``design_only`` or ``instanced`` flags are set
        """
        if not self._streaming:
            raise RuntimeError('Live data unavailable for instanced/design models')
        self.log.info('Starting background updates ...')
        self._interrupt = Event()
        _daemon = partial(Thread, daemon=True, target=self._background_update)
        self._daemons = [
            _daemon(args=(self._update_model, 'model-update')),
            _daemon(args=(self._update_LEM, 'LEM-watcher')),
            _daemon(args=(self._update_quads, 'acc1-watcher')),
            # _daemon(args=(self._update_misc, 'acc2-watcher')),
            ]
        for t in self._daemons: t.start()
        self.log.info('Online.')

    def stop(self):
        """ stop background processes """
        self._interrupt.set()
        for t in self._daemons: t.join()
        self.log.info('Background updates stopped.')

    def write_bmad(self, path=None):
        """
        save current lattice to a .bmad file, default filename is ``f2_elec_<ymdhms>.bmad``

        :param path: absolute filepath for desired output file
        """
        if not path:
            outdir = os.path.join(CONFIG['dirs']['model_data'], 'saved_lattices')
            fname = f"f2_elec_{datetime.today().strftime(CONFIG['ts_fmt'])}.bmad"
            path = os.path.join(outdir, fname)
        self.tao.cmd(f'write bmad -format one_file {path}')
        self.log.info(f'Lattice data written to {path}')

    def refresh_all(self, catch_errs=False):
        """
        single-shot model update (only for use with instanced models)

        :param catch_errs: catch errors and log during update rather than halt, defaults to False

        :raises RuntimeError: if the ``design_only`` flag is set, or ``instanced`` flag is not set
        """
        if self._streaming: raise RuntimeError('refresh_all only usable in instanced mode')
        self.log.info('Updating with live data...')
        tasks = [
            Thread(target=self._update_LEM),
            Thread(target=self._update_quads),
            # Thread(target=self._update_misc),
            ]
        try:
            t_st = time.time()
            for th in tasks: th.start()
            for th in tasks: th.join()
            self._update_model()
            self.log.info(f'Model refreshed in {time.time() - t_st:.3f}s')
        except Exception as err:
            msg = f'Model data refresh failed ({repr(err)})'
            if catch_errs: self.log.warnings(msg)
            else:
                self.log.critical(msg)
                raise err

    @property
    def tao(self):
        """ local instance of pytao.Tao """
        return self._tao
    
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
        valid masks are: ``RF, SOLN, XCOR, YCOR, COR, BEND, QUAD, SEXT, DRIFT, BPMS, PROF, DRIFT``

        :note: mask indicies are equivalent to: ``np.where(self.ele_types == '<Bmad ele.key>')``
        """
        return self._ix

    @property
    @cache
    def elements(self):
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
    def Z(self):
        """ linac Z position (floor coordinate) of all elements """
        return self._Z

    @property
    @cache
    def L(self):
        """ length of all elements in s-order """
        return self._lat_list_array('ele.l')

    @property
    @cache
    def device_names(self):
        """ control system channel access addresse of all elements in s-order """
        return self._device_names
        
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
        ``live.p0c, e_tot, gamma_rel, Brho, twiss, rf, quads, bends``

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

        :note: single-element transfer maps are calculated between the upstream and downstream
            faces of the element in question, while A-to-B transfer maps are calculated from
            the downstream face of element A and the downstream face of element B

        :param ele: beamline element(s), may be a single element e or a tuple of (e1, e2)
        :param which: which lattice to read from, default is 'model', can also choose 'design'

        :return: (R,v0) tuple of the map R (6x6 np.ndarray), and "0th order" map v0 (1x6 vector)
        """
        if type(ele) in [tuple, list]:
            e1, e2 = ele[0], ele[1]
        else:
            e1, e2 = ele, None

        if not e2 or e1 == e2:
            r = self.tao.ele_mat6(e1, which=which)
            v0 = self.tao.ele_mat6(e1,  which=which, who='vec0')
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

    def _background_update(self, target_fcn, name):
        # wrapper to run 'target_fcn' repeatedly until interrupted
        id_str = f'[{name}@{get_native_id()}]'
        while not self._interrupt.wait(CONFIG['bmad']['poll_interval']):
            try:
                t_st = time.time()
                target_fcn()
                self.log.debug(f'{id_str} iteration completed in {time.time()-t_st:.3f}s')
            except Exception as err:
                self.log.warn(f'{id_str} iteration FAILED: {repr(err)}')
        else:
            self.log.info(f'{id_str} Received interrupt signal, updating stopped.')

    def _update_model(self):
        # runs all commands submitted self._model_update_queue
        # and updates model data and device data accordingly
        N_update_dev = self._model_update_queue.qsize()
        N_update_lem = self._lem_update_queue.qsize()
        if (not N_update_dev) and (not N_update_lem): return

        # unload & execute all the updates in the queue
        device_updates = [self._model_update_queue.get_nowait() for _ in range(N_update_dev)]
        self.tao.cmds([
            f'set ele {name} {attr} = {value:.9f}' for (name, attr, value) in device_updates
            ])

        # update arrays & device dictionaries
        self._live_model_data.p0c   = self._lat_list_array('ele.p0c')
        self._live_model_data.e_tot = self._lat_list_array('ele.e_tot')
        self._live_model_data.twiss = self._fetch_twiss(which='model')
        for (name, attr, value) in device_updates:
            setattr(self._live_model_data.devices[name], attr, value)

        # update LEM data
        for reg in self.LEM:
            for i, elem in enumerate(reg.elements):
                i_global = self.ix[elem]
                reg.EACT[i] = self.live.p0c[i_global]
                reg.BLEM[i] = self.live.Brho[i_global] * self.design.quads[elem].k1 * reg.L[i]

        # grab any new amplitude/chirp/fudge numbers from their queue
        for _ in range(N_update_lem):
            (attr, vals) = self._lem_update_queue.get_nowait()
            setattr(self.LEM.L0, attr, vals[0])
            setattr(self.LEM.L1, attr, vals[1])
            setattr(self.LEM.L2, attr, vals[2])
            setattr(self.LEM.L3, attr, vals[3])

    def _update_LEM(self):
        # updates the live momentum profile and LEM data
        V_acts, phases, sbst_phases, fudges, amplitudes, chirps = self._calc_live_pz()
        self._set_cavities(V_acts, phases, sbst_phases, fudges)
        self._lem_update_queue.put(('amplitude', amplitudes))
        self._lem_update_queue.put(('chirp', chirps))
        self._lem_update_queue.put(('fudge', fudges))

    def _calc_live_pz(self):
        # calculates the live beam momentum profile p_z(s) from the current klystron complement
        # as well as per-linac fudge/Egain/chirp values

        klys_status = slc.get_all_klys_stat()

        # TODO: get expected amplitudes from bend magnet settings, not design model
        p0c_l0 = self.design.p0c[self.ix[CONFIG['linac']['L0']['e_end']]]
        p0c_l1 = self.design.p0c[self.ix[CONFIG['linac']['L1']['e_end']]]
        p0c_l2 = self.design.p0c[self.ix[CONFIG['linac']['L2']['e_end']]]
        p0c_l3 = self.design.p0c[self.ix[CONFIG['linac']['L3']['e_end']]]
        ampl_design = [
            p0c_l0,
            p0c_l1 - p0c_l0,
            p0c_l2 - p0c_l1,
            p0c_l3 - p0c_l2,
            ]

        # calculate fudge factors for each linac (currently faking L0, L1)
        # TODO: get L0, L1 for real
        ampl_est = [0, 0, 0, 0]
        chirp_est = [0, 0, 0, 0]
        ampl_est[:2] = ampl_design[:2]

        # get all klystron ENLDs, phases & on/off stats to estimate momentum profile
        V_acts, phases, enables, sbst_phases = {}, {}, {}, {}
        for kname, cavities in self.klys_structure_map.items():
            if kname in CONFIG['linac']['bad_klys']: continue

            k_ch = self._klys_channels[kname]
            if kname == 'K13_2': k_ch = 'LI13:KLYS:21'
            ch_parts = k_ch.split(':')
            sector = int(ch_parts[0][-2:])
            # alternate channel address since AIDA-PVA flips micro/primary
            k_ch_alt = f'{ch_parts[1]}:{ch_parts[0]}:{ch_parts[2]}'
            linac = 2 if sector < 15 else 3

            if sector not in sbst_phases.keys():
                sbst_phases[sector] = get_pv(f'LI{sector}:SBST:1:PDES').value
            enables[kname] = 1 if klys_status[k_ch_alt]['accel'] else 0
            ENLD = get_pv(f'{k_ch}:ENLD').value * 1e6
            phases[kname] = get_pv(f'{k_ch}:PDES').value

            # if a klystron is not on-beam, just set cavitity amplitudes to 0
            V_acts[kname] = enables[kname] * ENLD
            phi = np.deg2rad(phases[kname] + sbst_phases[sector])
            ampl_est[linac] = ampl_est[linac] + V_acts[kname] * np.cos(phi)
            chirp_est[linac] = chirp_est[linac] + V_acts[kname] * np.sin(phi)

        # calculate per-linac fudges
        fudges = [ampl_design[i] / ampl_est[i] for i in range(4)]

        return V_acts, phases, sbst_phases, fudges, ampl_est, chirp_est

    def _set_cavities(self, V_acts, phases, sbst_phases, fudges):
        # set cavity amplitudes & phases according to the input momentum profile
        # data & fudge values from self._calc_live_pz
        for kname, cavities in self.klys_structure_map.items():
            if kname in CONFIG['linac']['bad_klys']:
                for cav in cavities: self._model_update_queue.put((cav, 'voltage', 0.0))
                continue

            k = int(kname[1:3])
            sector = int(self._klys_channels[kname].split(':')[0][-2:])
            if sector == 11 and k < 3: linac = 1
            elif sector < 15:  linac = 2
            elif sector >= 15: linac = 3

            # assume that power is distributed evenly to each DLWG
            V_cavity = fudges[linac] * (V_acts[kname] / len(cavities))
            phi_cavity = sbst_phases[sector] + phases[kname]

            for cav in cavities:
                self._model_update_queue.put((cav, 'voltage', V_cavity))
                self._model_update_queue.put((cav, 'phi0', phi_cavity/360.0))

    def _update_quads(self, var='BDES'):
        # updates all quadrupole magnets in the accelerator
        for qname in self.elements[self._ix['QUAD']]:
            q_ch = self._device_names[self._ix[qname]]
            if q_ch == '': continue

            # some devices have both a main a secondary ("boost") power supply
            bact = get_pv(f'{q_ch}:{var}').value
            boost_bact = 0.0
            if q_ch in self._secondary_devices.keys():
                boost_bact = get_pv(f'{self._secondary_devices[q_ch]}:{var}').value
            
            grad = intkGm_2_gradTm(bact + boost_bact, self.L[self._ix[qname]])
            self._model_update_queue.put((qname, 'b1_gradient', grad))

    def _update_misc(self, var='BDES'):
        # updates bend magnets and other assorted devices
        for bname in self.elements[self._ix['BEND']]:
            ch = self._device_names[self._ix[bname]]
            if ch == '': continue

            # TO DO: figure out what param to set (rho, fint ...)
            print(f'\n{bname} ({ch})')
            B = get_pv(f'{ch}:{var}').value
            b1 = get_pv(f'{ch}:{var}').value
            b2 = self.design.bends[bname].g 
            grad = intkGm_2_gradTm(B, self.L[self._ix[bname]])
            print(f'{b1:.4f} {b2:.4f} {grad:.4f}')
            self._model_update_queue.put((bname, 'b_field', grad))

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
    
    def _init_static_lattice_data(self):
        # loads in static params: element names, positions and lord/slave config

        # make a list of all cavities in each linac
        self._cav_l0 = self._lat_list_array('ele.name', elems=CONFIG['linac']['L0']['knames'])
        self._cav_l1 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CONFIG['linac']['L1']['knames']]
            )
        self._cav_l2 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CONFIG['linac']['L2']['knames']]
            )
        self._cav_l3 = np.concatenate(
            [self._lat_list_array('ele.name', elems=cs) for cs in CONFIG['linac']['L3']['knames']]
            )
        self._all_cavs = np.concatenate([self._cav_l0, self._cav_l1, self._cav_l2, self._cav_l3])

        # map of L1 - L3 klystron names and their associated cavities
        # Bmad only deals in individual cavities, so we need this metadata
        # L0A/B are single structures, and so not included
        self._linac_klys = []
        for cav_name in np.concatenate([self._cav_l2, self._cav_l3]):
            k_id = cav_name[:5]
            if k_id not in self._linac_klys: self._linac_klys.append(k_id)
        self.klys_structure_map = {}
        for k_id in self._linac_klys:
            self.klys_structure_map[k_id] = self._lat_list_array('ele.name', elems=f'{k_id}*')

        # the self.ix dictionary provides an interface for array indexing
        # contains indices of single elements, and index masks for element types
        self._ix = {}
        for i, e in enumerate(self.elements): self._ix[e] = i

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
        for ix, e in enumerate(self.elements):
            if self.ele_types[ix] != 'Monitor': continue
            if e[:3] == 'BPM' or e[0] == 'M': self._ix['BPMS'].append(ix)
            else: self._ix['PROF'].append(ix)

        # get control system names (stored as the element "alias" in Bmad)
        self._device_names = np.empty(self.elements.size, dtype='<U20')
        self._klys_channels = {}
        for line in self.tao.show('lattice -attr alias -all -no_slaves -no_label_lines -python'):
            ele_name = line.split(';')[1]
            ch_name = line.split(';')[-1]
            if ch_name == '': continue
            idx = self.ix[ele_name]
            self._device_names[idx] = ch_name
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

        # (TEMPORARY)
        # some elements are missing control system names (alias) in Bmad,
        # load them from a text file instead
        with open('config/unaliased-elements.csv', 'r') as f:
            ldata = [l.split(',') for l in f.readlines()[1:]]
        self._secondary_devices = {}
        for l in ldata:
            elem, etype, dev, dev2 = l[0], l[1], l[2], l[3].strip()
            idx = self._ix[elem]
            self._device_names[idx] = dev
            self._ix[dev] = idx
            if dev2 != '': self._secondary_devices[dev] = dev2

        # some quads, cavities and other elements are split into multiple parts
        # keep a dictionary of the slave elements for use later
        # also disabling debug messages here -- too much spam for even 'debug' mode
        self.log.debug('Tagging slave elements ...')
        logging.disable(logging.DEBUG)
        self._slave_rf, self._slave_quads, self._slave_misc = {}, {}, {}
        for ele, etype in zip(self.elements, self.ele_types):
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

        # read linac Z positions from floor coordinates
        self._Z = np.empty(self.elements.size)
        for i, e in enumerate(self.elements):
            self._Z[i] = self.tao.ele_floor(i)['Reference'][2]

    def _init_static_LEM_data(self):
        # initialzes _LEMRegionData for L0 - L3
        regions = []
        for rname in CONFIG['linac']['LEM_regions']:
            e_start, e_end = CONFIG['linac'][rname]['e_start'], CONFIG['linac'][rname]['e_end']
            i_start, i_end = self.ix[e_start], self.ix[e_end]

            # grab all the quads in this region
            region_elems = []
            for ixr, ele in enumerate(self.elements[i_start:i_end]):
                etype = self.ele_types[ixr + i_start]
                if self.ele_types[ixr + i_start] == 'Quadrupole':
                    if ele[:2] in ['CQ','SQ']: continue
                    region_elems.append(ele)

            # initialize region data & populate static data
            reg = _LEMRegionData(len(region_elems))
            for i, ele in enumerate(region_elems):
                i_global = self.ix[ele]
                reg.elements[i] = ele
                reg.device_names[i] = self.device_names[i_global]
                reg.S[i] = self.S[i_global]
                reg.Z[i] = self.Z[i_global]
                reg.L[i] = self.L[i_global]
                reg.EREF[i] = self.design.p0c[i_global]
            regions.append(reg)

        self.LEM = _F2LEMData(
            L0=regions[0], L1=regions[1], L2=regions[2], L3=regions[3],
            )

    def _init_static_device_data(self):
        # get design device settings from Bmad, these device dictionaries are
        # also duplicated to initialize the live _ModelData struct
        _req_cav = partial(self._lat_list_array, elems='lcavity::*')
        _req_bend = partial(self._lat_list_array, elems='sbend::*')
        _req_quad = partial(self._lat_list_array, elems='quad::*')

        for n,s,l,v,phi in zip(
            _req_cav('ele.name'),_req_cav('ele.s'),_req_cav('ele.l'),
            _req_cav('ele.voltage'),_req_cav('ele.phi0'),
            ):
            # only including the forward RF for now. Also L1XF doesn't exist...
            if n in ['TCY10490', 'L1XF', 'TCY15280']: continue
            self._design_model_data.rf[n] = _Cavity(S=s, l=l, voltage=v, phase=360.*phi)

        for n,s,l,b,g in zip(
            _req_bend('ele.name'),_req_bend('ele.s'),_req_bend('ele.l'),
            _req_bend('ele.b_field'),_req_bend('ele.g'),
            ):
            self._design_model_data.bends[n] = _Dipole(S=s, l=l, b_field=b, g=g)

        for n,s,l,b,k1 in zip(
            _req_quad('ele.name'),_req_quad('ele.s'),_req_quad('ele.l'),
            _req_quad('ele.b1_gradient'),_req_quad('ele.k1'),
            ):
            self._design_model_data.quads[n] = _Quad(S=s, l=l, b1_gradient=b, k1=k1)

    def _lat_list_array(self, who, elems='*', which='model', dtype=np.float64):
        # packs a single-column of lattice data from tao.lat_list as an array
        if dtype is np.float64:
            return np.array(
                self.tao.lat_list(elems, who, which=which, flags='-array_out -no_slaves')
                )
        return np.array(
            self.tao.lat_list(elems, who, which=which, flags='-no_slaves')
            ).astype(dtype)
