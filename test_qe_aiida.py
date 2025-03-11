#!/usr/bin/env python
from aiida import load_profile
load_profile()

from aiida import orm
from aiida.engine import submit
from aiida.plugins import CalculationFactory, GroupFactory

# Load code
code = orm.load_code('pw-local@localhost')

# Structure
alat = 5.431
cell = [[alat/2, alat/2, 0], [alat/2, 0, alat/2], [0, alat/2, alat/2]]
structure = orm.StructureData(cell=cell)
structure.append_atom(position=(0., 0., 0.), symbols='Si')
structure.append_atom(position=(alat/4, alat/4, alat/4), symbols='Si')

# K-points
kpoints = orm.KpointsData()
kpoints.set_kpoints_mesh([2, 2, 2])

# Parameters for relax calculation
parameters = orm.Dict(
    dict={
        'CONTROL': {
            'calculation': 'vc-relax',  # variable-cell relaxation
            'restart_mode': 'from_scratch',
            'tstress': True,
            'tprnfor': True,
        },
        'SYSTEM': {
            'ecutwfc': 30.,
            'ecutrho': 240.,
        },
        'ELECTRONS': {
            'conv_thr': 1.e-8,
        },
        'IONS': {
            'ion_dynamics': 'bfgs',
        },
        'CELL': {
            'cell_dynamics': 'bfgs',
        }
    }
)

# Get pseudopotentials
SsspFamily = GroupFactory('pseudo.family.sssp')
pseudo_family = SsspFamily.collection.get(label='SSSP/1.3/PBE/efficiency')
print(f"Using pseudopotential family: {pseudo_family.label}")

# Create calculation
PwCalculation = CalculationFactory('quantumespresso.pw') #Here it uses aiida-quantumespresso plugin
builder = PwCalculation.get_builder()

# Configure inputs
builder.structure = structure
builder.kpoints = kpoints
builder.parameters = parameters
builder.code = code
builder.pseudos = pseudo_family.get_pseudos(structure=structure)

# Resources
builder.metadata.options = {
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
    'max_wallclock_seconds': 1800,
    'withmpi': True,
}

builder.metadata.description = 'Silicon relaxation using QE pw.x'
builder.metadata.label = 'Si_relax'

# Submit
node = submit(builder)
print(f"Submitted calculation with pk: {node.pk}")
print(f"Monitor with: verdi process show {node.pk}")