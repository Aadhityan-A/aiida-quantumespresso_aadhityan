from aiida import load_profile
from aiida.orm import QueryBuilder, Group, load_node
from aiida.plugins import DataFactory

# Load the AiiDA profile
load_profile()

# Get the UpfData class
UpfData = DataFactory('core.upf')

# List all pseudopotential families
def list_pseudo_families():
    """List all pseudopotential families in the database."""
    qb = QueryBuilder()
    qb.append(Group, filters={'type_string': 'core.upf.family'})
    
    print("Available pseudopotential families:")
    for group in qb.all(flat=True):
        # Fixed indentation here
        print(f"- {group.label}: {group.description}")
        # Count pseudopotentials in this family
        count = len(group.nodes)
        print(f"  Contains {count} pseudopotentials")
        
        # List elements in this family
        elements = set()
        for upf in group.nodes:
            # Fixed indentation here
            elements.add(upf.element)
        print(f"  Elements: {', '.join(sorted(elements))}")
    
    return qb.all(flat=True)

# Get pseudopotentials for a specific element from a family
def get_pseudos_by_element(element, family_name):
    """Get pseudopotentials for a specific element from a family."""
    qb = QueryBuilder()
    qb.append(Group, filters={'label': family_name, 'type_string': 'core.upf.family'}, tag='group')
    qb.append(UpfData, with_group='group', filters={'attributes.element': element})
    
    results = qb.all(flat=True)
    print(f"Found {len(results)} pseudopotentials for element {element} in family {family_name}")
    
    for pseudo in results:
        # Fixed indentation here
        print(f"- {pseudo.filename}: UUID={pseudo.uuid}")
    
    return results

# Example usage
list_pseudo_families()

# If you want to check pseudos for a specific element in a family
# Uncomment and modify the line below
# get_pseudos_by_element('Si', 'SSSP_1.1_efficiency')