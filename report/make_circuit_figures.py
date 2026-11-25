# L-06: circuit figures of the report. Needs H-02 merged (store hw-20261125T124523Z-main-HW) and pylatexenc for the drawer.
# usage: python report/make_circuit_figures.py
import json
import sys

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit.visualization import plot_circuit_layout  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston  # noqa: E402

from quantum.feature_map import feature_map  # noqa: E402
from scripts.qpu_campaign import gravityspy_features  # noqa: E402

OUT, RUN = 'report/fig/', 'store/runs/hw-20261125T124523Z-main-HW/'
STYLE = {'name': 'clifford', 'backgroundcolor': '#FDFDFB', 'fontsize': 13, 'subfontsize': 9}
NAMES = ['SNR', 'f picco', 'f centr', 'banda', 'durata', 'Q', 'ampiezza']  # FEATURE_COLUMNS order

qc = feature_map(8, 2)
named = qc.assign_parameters({p: Parameter('θ ' + n) for p, n in zip(qc.parameters, NAMES)})
named.draw('mpl', style=STYLE, fold=-1, scale=0.9).savefig(OUT + 'feature_map_named.png', dpi=200, bbox_inches='tight')

meta, ids = json.load(open(RUN + 'meta.json')), json.load(open(RUN + 'ids.json'))
frozen = yaml.safe_load(open('configs/hardware_frozen.yaml'))
angles = frozen['s'] * np.pi * gravityspy_features(frozen)(ids[:1])[0]
dev = FakeKingston()
# shortcut: FakeKingston's snapshot picks other qubits, so the measured layout is imposed
isa = generate_preset_pass_manager(optimization_level=3, backend=dev, seed_transpiler=7,
                                   initial_layout=meta['layout']).run(qc.assign_parameters(angles[:7]))
isa.draw('mpl', style=STYLE, idle_wires=False, fold=-1, scale=0.7).savefig(OUT + 'transpiled.png', dpi=170, bbox_inches='tight')
plot_circuit_layout(isa, dev, view='physical').savefig(OUT + 'layout.png', dpi=60, bbox_inches='tight')
plt.close('all')
print(dict(isa.count_ops()), isa.depth(), isa.depth(lambda i: i.operation.num_qubits == 2))
