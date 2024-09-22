====
FACET-II Live Model Infrastructure
====
Code for simulating the production FACET accelerator with the FACET2E Bmad model. There are two primary layers of infrastructure here:
* **BmadLiveModel**: code monitoring the production controls system and updating a local instance of PyTao to simulate the FACET-II linac
* **FACET live model PVA server**: a python PVA server to publish various model data to PVs

BmadLiveModel
-------
A python class for monitoring the live accelerator controls system via EPICS Channel Access and AIDA-PVA as needed, and updating a local instance of PyTao to match extant machine settings. The live model is also responsible for assessing the current klystron complement and estimating the live momentum profile of the beam, as well as suggested lattice magnet settings BLEM.
* Link to full design overview: https://confluence.slac.stanford.edu/display/FACET/Bmad+live+modeling
* API documentation: https://f2-live-model.readthedocs.io/en/latest/
stuff
* Built using pyTao: https://bmad-sim.github.io/pytao/

The simplest method of using this class is to only serve the design model. This is basically like using PyTao with extra steps.
>>> from F2_live_model.bmad import BmadLiveModel
>>> f2m = BmadLiveModel(design_only=True)

There are two methods of actually monitoring the production accelerator, the first of which is to update data only when asked. This is called "instanced" mode.
>>> from F2_live_model.bmad import BmadLiveModel
>>> f2m = BmadLiveModel(instanced=True)
>>> # your code goes here
>>> f2m.refresh_all()
>>> # even more code!

The second method is to stream data in real-time, which is more resource intensive. This can be handled either by using a contexet manager (preferred) or with the provided .start and .stop functions.
>>> from F2_live_model.bmad import BmadLiveModel
>>> with BmadLiveModel() as f2m:
>>>    # your code here, updates happen in the background

>>> from F2_live_model.bmad import BmadLiveModel
>>> f2m = BmadLiveModel()
>>> f2m.start()
>>> # your code here, updates happen in the background
>>> f2m.stop()

The BmadLiveModel object stores some live & design model data locally as python data structures.
>>> f2m.design.twiss.beta_x
<np.ndarray>
>>> f2m.live.p0c
<np.ndarray>
>>> f2m.get_rmat('QE10525')
<6x6 np.ndarray>

The local instance of PyTao is also publicly accessible.
>>> f2m.tao.cmd('set ele QE10525 b1_gradient -1.5')
<results snipped>

Full interface details can be found in the online documentation.

FACET Live Model PVA server
-------
The FACET model server is responsible for running a live-streaming BmadLiveModel, and publishing data from it to PVs via PVA. The provided RMAT and twiss tables are compatible with the python meme service.
* Documentation for the meme service is here: https://github.com/slaclab/meme

The live model server also published LEM data, which is not ingested by meme, but is used by the LEM server app (coming soon ...).

This server published the following PVs:
BMAD:SYS0:1:FACET2E:DESIGN:TWISS
BMAD:SYS0:1:FACET2E:DESIGN:RMAT
BMAD:SYS0:1:FACET2E:DESIGN:URMAT
BMAD:SYS0:1:FACET2E:LIVE:TWISS
BMAD:SYS0:1:FACET2E:LIVE:RMAT
BMAD:SYS0:1:FACET2E:LIVE:URMAT
BMAD:SYS0:1:FACET2E:LEM:DATA
BMAD:SYS0:1:FACET2E:LEM:L0_AMPL
BMAD:SYS0:1:FACET2E:LEM:L1_AMPL
BMAD:SYS0:1:FACET2E:LEM:L2_AMPL
BMAD:SYS0:1:FACET2E:LEM:L3_AMPL
BMAD:SYS0:1:FACET2E:LEM:L0_CHIRP
BMAD:SYS0:1:FACET2E:LEM:L1_CHIRP
BMAD:SYS0:1:FACET2E:LEM:L2_CHIRP
BMAD:SYS0:1:FACET2E:LEM:L3_CHIRP
BMAD:SYS0:1:FACET2E:LEM:L0_FUDGE
BMAD:SYS0:1:FACET2E:LEM:L1_FUDGE
BMAD:SYS0:1:FACET2E:LEM:L2_FUDGE
BMAD:SYS0:1:FACET2E:LEM:L3_FUDGE

