#!/usr/bin/env python
# scf_calculation.py

import os
import numpy as np
from aiida import load_profile
from aiida.engine import submit
from aiida.orm import Code, Dict, StructureData, KpointsData
from aiida_quantumespresso.calculations.pw import PwCalculation

# Load the AiiDA profile
load_profile()

# Configure the Quantum ESPRESSO pw.x code if not already configured
def setup_code():
    """Set up the pw.x code if it doesn't exist yet."""
    from aiida.orm import Code
    from aiida.common.exceptions import NotExistent
    
    code_label = 'pw-qe'
    try:
        code = Code.get_from_string(f'{code_label}@localhost')
        print(f"Code '{code_label}' is already configured.")
        return code
    except NotExistent:
        print(f"Code '{code_label}' not found. Configuring it now...")
        
        # Assuming pw.x is in the PATH, otherwise specify the full path
        code = Code(
            input_plugin_name='quantumespresso.pw',
            remote_computer_exec=[
                ('login.mesu.sorbonne-universite.fr', '/home/arivazha/gitlab/q-e-pioud/bin/pw.x')  # Replace with actual path to pw.x
            ]
        )
        code.label = code_label
        code.description = 'MeSU Quantum ESPRESSO pw.x code'
        code.store()
        
        print(f"Code '{code_label}' configured successfully.")
        return code

# Define the structure (Silicon)
def create_silicon_structure():
    """Create a silicon crystal structure."""
    alat = 5.43  # Lattice parameter in Angstrom
    cell = [[alat/2, alat/2, 0], [alat/2, 0, alat/2], [0, alat/2, alat/2]]
    
    structure = StructureData(cell=cell)
    structure.append_atom(position=(0., 0., 0.), symbols='Si')
    structure.append_atom(position=(alat/4, alat/4, alat/4), symbols='Si')
    
    return structure

# Define the k-points mesh
def create_kpoints():
    """Create a k-points mesh."""
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 4])
    return kpoints

# Define the calculation parameters
def create_parameters():
    """Create the parameters for the calculation."""
    parameters = Dict(
        dict={
            'CONTROL': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'wf_collect': True,
                'tstress': True,
                'tprnfor': True,
            },
            'SYSTEM': {
                'ecutwfc': 30.,
                'ecutrho': 240.,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.1,
            },
            'ELECTRONS': {
                'conv_thr': 1.e-8,
                'mixing_beta': 0.7,
                'electron_maxstep': 100,
            }
        }
    )
    return parameters

# Set up pseudopotentials
def setup_pseudos():
    """Set up pseudopotentials for the calculation."""
    import os
    from aiida.orm import UpfData, Group
    from aiida.common.exceptions import NotExistent
    
    # Check if we already have a pseudopotential family
    family_name = 'Si_pseudo'
    try:
        group = Group.get(label=family_name)
        print(f"Pseudopotential family '{family_name}' already exists.")
    except NotExistent:
        print(f"Creating pseudopotential family '{family_name}'...")
        group = Group(label=family_name)
        group.store()
        
        # Create a Si pseudopotential (you would normally download this)
        # For this example, we'll assume you have a Si.upf file
        pseudo_dir = os.path.expanduser('~/pseudo')
        os.makedirs(pseudo_dir, exist_ok=True)
        
        # You need to download or provide a Si.upf file
        # For example, you can download it from Quantum ESPRESSO website
        # and place it in the pseudo_dir
        
        # Check if the pseudopotential file exists
        pseudo_path = os.path.join(pseudo_dir, 'Si.upf')
        if not os.path.exists(pseudo_path):
            print(f"Please download a Si.upf file and place it in {pseudo_dir}")
            print("You can download it from: https://www.quantum-espresso.org/pseudopotentials/")
            raise FileNotFoundError(f"Si.upf not found in {pseudo_dir}")
        
        # Create the UpfData node
        upf = UpfData(file=pseudo_path)
        upf.store()
        
        # Add it to the group
        group.add_nodes([upf])
        print(f"Added Si pseudopotential to family '{family_name}'")
    
    # Get the pseudos for our structure
    structure = create_silicon_structure()
    pseudos = {}
    for kind in structure.get_kind_names():
        pseudo = UpfData.get_upf_node(element=kind, family_name=family_name)
        pseudos[kind] = pseudo
    
    return pseudos

def main():
    """Set up and submit a PW calculation."""
    # Set up the code
    code = setup_code()
    
    # Create the calculation process
    builder = PwCalculation.get_builder()
    
    # Set the required inputs
    builder.code = code
    builder.structure = create_silicon_structure()
    builder.kpoints = create_kpoints()
    builder.parameters = create_parameters()
    
    # Set up pseudopotentials
    builder.pseudos = setup_pseudos()
    
    # Set the parallelization options
    builder.metadata.options.resources = {
        'num_machines': 1,
        'num_mpiprocs_per_machine': 4,
    }
    
    # Set the maximum wallclock time
    builder.metadata.options.max_wallclock_seconds = 3600  # 1 hour
    
    # Set the queue name (if needed)
    # builder.metadata.options.queue_name = 'your_queue'
    
    # Submit the calculation
    print("Submitting calculation...")
    calc = submit(builder)
    print(f"Submitted calculation with pk: {calc.pk}")
    print(f"You can check its progress with: verdi process list -a")
    print(f"And get more details with: verdi process show {calc.pk}")

if __name__ == "__main__":
    main()