#!/usr/bin/env python
from aiida import load_profile
load_profile()

from aiida import orm
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the calculation node
calc_pk = 112
calc = orm.load_node(calc_pk)

print(f"Calculation {calc_pk}: {calc.process_label} [{calc.process_state}]")
print(f"Description: {calc.description}")
print(f"Created: {calc.ctime.strftime('%Y-%m-%d %H:%M:%S')}")

# Check inputs
print("\n=== INPUTS ===")
print(f"Code: {calc.inputs.code.label} on {calc.inputs.code.computer.label}")
print(f"k-points mesh: {calc.inputs.kpoints.get_kpoints_mesh()[0]}")
print(f"Calculation type: {calc.inputs.parameters.get_dict()['CONTROL']['calculation']}")

# Get pseudopotential information properly
print("Pseudopotentials:")
for element, pseudo in calc.inputs.pseudos.items():
    if hasattr(pseudo, 'filename'):
        print(f"  {element}: {pseudo.filename}")
    else:
        print(f"  {element}: {pseudo}")

# Check if calculation finished successfully
# if calc.exit_status == 0:
#     print("\n=== OUTPUTS ===")

    # Get the relaxed structure
if 'output_structure' in calc.outputs:
    structure = calc.outputs.output_structure
    print("\nStructure:")
    print(f"  Cell parameters (Å): {[round(x, 6) for x in structure.cell_lengths]}")
    print(f"  Cell volume (Å³): {round(structure.get_cell_volume(), 6)}")
    print(f"  Number of atoms: {len(structure.sites)}")

    # Export structure to multiple formats
    for fmt in ['xsf', 'cif']:
        filename = f"relaxed_structure_{calc_pk}.{fmt}"
        structure.export(filename)
        print(f"  Exported to: {filename}")

# Get detailed output parameters
if 'output_parameters' in calc.outputs:
    params = calc.outputs.output_parameters.get_dict()

    print("\nOutput Parameters:")
    print(f"  Available keys: {list(params.keys())}")

    # Print energy information
    print("\nEnergy Information:")
    energy_keys = {k: v for k, v in params.items() if 'energy' in k.lower()}
    for key, value in energy_keys.items():
        print(f"  {key}: {value}")

    # Print forces if available
    if 'forces' in params:
        forces = np.array(params['forces'])
        max_force = np.max(np.abs(forces))
        print(f"\nForces:")
        print(f"  Max force component: {max_force:.6f} eV/Å")

    # Print stress if available
    if 'stress' in params:
        stress = np.array(params['stress'])
        print(f"\nStress tensor (GPa):")
        for i in range(min(3, len(stress))):
            if isinstance(stress[i], list) and len(stress[i]) >= 3:
                print(f"  {stress[i][0]:10.4f} {stress[i][1]:10.4f} {stress[i][2]:10.4f}")
            else:
                print(f"  {stress[i]}")

    # Print any available information
    print("\nOther Information:")
    for key, value in params.items():
        if key not in energy_keys and key not in ['forces', 'stress']:
            if not isinstance(value, (list, dict)) or len(str(value)) < 100:
                print(f"  {key}: {value}")

# Print retrieval instructions for manual inspection
print("\n=== MANUAL INSPECTION ===")
print("To view the calculation output manually:")
print(f"  verdi calcjob outputcat {calc_pk} | less")
print(f"  verdi calcjob inputcat {calc_pk} | less")
print(f"  verdi process show {calc_pk}")