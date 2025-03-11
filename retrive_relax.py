from aiida import load_profile
load_profile()
from aiida import orm

# Load the calculation node
calc = orm.load_node(112)

# Get the relaxed structure
if 'output_structure' in calc.outputs:
    relaxed_structure = calc.outputs.output_structure
    print(f"Final cell parameters: {relaxed_structure.cell_lengths}")
    print(f"Final cell volume: {relaxed_structure.get_cell_volume()} Å³")

# Get the final energy - first check what keys are available
if 'output_parameters' in calc.outputs:
    params = calc.outputs.output_parameters.get_dict()

    # Print available keys to identify the correct energy key
    print("\nAvailable output parameters:")
    for key in params.keys():
        print(f"- {key}")

    # Access energy with correct key name
    if 'energy' in params:
        print(f"\nFinal energy: {params['energy']} eV")
    elif 'energy_final' in params:
        print(f"\nFinal energy: {params['energy_final']} eV")
    elif 'total_energy' in params:
        print(f"\nFinal energy: {params['total_energy']} eV")
    else:
        # Common key in QE output
        print(f"\nFinal free energy: {params.get('energy_free', 'N/A')} eV")

    # Check if convergence was achieved
    if 'finished_ok' in params:
        print(f"Calculation finished successfully: {params['finished_ok']}")

    # Get forces and stresses if available
    if 'forces' in params:
        print(f"Final forces (max component): {max(abs(params['forces'].flatten()))}")
    if 'stress' in params:
        print(f"Final stress tensor: {params['stress']}")

# Export relaxed structure to a file
if 'output_structure' in calc.outputs:
    filename = "relaxed_si.xsf"
    relaxed_structure.export(filename)
    print(f"\nRelaxed structure exported to {filename}")