"""Utility functions for preprocessing before submission in LordCapulet.
"""

# default manifold for each atom

default_manifold = {
    # chalcogens use p orbitals
    'O': '2p',
    'S': '3p',
    'Se': '4p',
    'Te': '5p',
    # halogens use p orbitals
    'F': '2p',
    'Cl': '3p',
    'Br': '4p',
    'I': '5p',
    # transition metals use d orbitals
    'Sc': '3d',
    'Ti': '3d',
    'V': '3d',
    'Cr': '3d',
    'Mn': '3d',
    'Fe': '3d',
    'Co': '3d',
    'Ni': '3d',
    'Cu': '3d',
    'Zn': '3d',
    'Y': '4d',
    'Zr': '4d',
    'Nb': '4d',
    'Mo': '4d', 
    'Tc': '4d',
    'Ru': '4d',
    'Rh': '4d',
    'Pd': '4d',
    'Ag': '4d',
    'Cd': '4d',
    'La': '5d',
    'Hf': '5d',
    'Ta': '5d',
    'W': '5d',  
    'Re': '5d',
    'Os': '5d',
    'Ir': '5d',
    'Pt': '5d',
    'Au': '5d',
    'Hg': '5d',
    # the actinides use 5f orbitals
    'Ac': '5f',
    'Th': '5f',
    'Pa': '5f',
    'U': '5f',
    'Np': '5f',
    'Pu': '5f',
    'Am': '5f',
    # the lanthanides use 4f orbitals
    'Ce': '4f',
    'Pr': '4f',
    'Nd': '4f',
    'Pm': '4f',
    'Sm': '4f',
    'Eu': '4f',
    'Gd': '4f',
    'Tb': '4f',
    'Dy': '4f',
    'Ho': '4f',
    'Er': '4f', 
    'Tm': '4f',
    'Yb': '4f',
    'Lu': '4f',
}

# default dimensions for each manifold
default_dimensions = {
    's': 1,
    'p': 3,
    'd': 5,
    'f': 7,
}




#%%
def tag_and_list_atoms(atoms, table=None):
    """
    Tags atoms based on whether they are transition metals or other elements.
    Transition metals get a unique tag (e.g., Ni1, Mn2).
    Other elements get a tag based on their element symbol (e.g., O1, S1).
    These tags are stored in atom.info['custom_tag'].

    Args:
        atoms (list): A list of atom objects, assumed to be ASE Atom objects
                      or similar with 'symbol' and an 'info' dictionary attribute.
    """

    if table is None:
        table = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'U',
        }
    else:
        # assert that table is a set of strings
        assert isinstance(table, set), "table must be a set of element symbols as strings."

        for el in table:
            if not isinstance(el, str):
                raise ValueError("table must be a set of element symbols as strings.")

    tm_counts = {}
    other_counts = {}
    tm_atoms = []

    for atom in atoms:

        if atom.symbol in table:
            if atom.symbol not in tm_counts:
                tm_counts[atom.symbol] = 0
            
            tm_counts[atom.symbol] += 1
            # Store the custom string tag in atom.info
            atom.tag = tm_counts[atom.symbol]
            tm_atoms.append(f"{atom.symbol}{tm_counts[atom.symbol]}")
        else:
            if atom.symbol not in other_counts:
                other_counts[atom.symbol] = 1
            
            # Store the custom string tag in atom.info
            atom.tag = other_counts[atom.symbol]
    
    return tm_atoms

# function that gets a tm_atoms list and returns a list of default manifolds
def get_default_manifolds(tm_atoms):
    """
    Given a list of tagged transition metal atoms (e.g., ['Fe1', 'Ni1', 'Fe2']),
    returns a list of their default manifolds based on the predefined mapping.

    Args:
        tm_atoms (list): A list of tagged transition metal atom strings.

    Returns:
        list: A list of default manifolds corresponding to the input atoms.
    """
    manifolds = []
    for tm_atom in tm_atoms:
        element = ''.join(filter(str.isalpha, tm_atom))  # Extract element symbol
        manifold = default_manifold.get(element)
        if manifold is None:
            raise ValueError(f"No default manifold found for element: {element}")
        manifolds.append(manifold)
    
    return manifolds


# calculate dimensions for each manifold
# this is manifold_dim x manifold_dim x 2 to account for spin

def get_dimensions(manifolds):
    """
    Given a list of manifolds (e.g., ['3d', '4f', '3d']),
    returns a list of their corresponding dimensions based on the predefined mapping.

    Args:
        manifolds (list): A list of manifold strings.

    Returns:
        list: A list of dimensions corresponding to the input manifolds.
    """
    dimensions = []
    for manifold in manifolds:
        orbital_type = manifold[1]  # e.g., 'd' from '3d'
        dim = default_dimensions.get(orbital_type)
        if dim is None:
            raise ValueError(f"No default dimension found for manifold: {manifold}")
        dimensions.append(dim * dim * 2)  # account for spin
    
    return dimensions


