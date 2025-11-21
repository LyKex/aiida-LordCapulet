
import numpy as np
import json
import contextlib
import io
from aiida.orm import Dict, Code, KpointsData, load_node, JsonableData
from aiida.engine import WorkChain, run
from aiida.orm import Dict, List, Int, Float, Str
from aiida.engine import calcfunction

from .proposal_modes import propose_random_constraints, propose_random_so_n_constraints
from lordcapulet.utils import OccupationMatrixData, OccupationMatrixAiidaData, extract_occupations_from_calc, filter_atoms_by_species


def redirect_print_report(func, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    output = buf.getvalue()
    return result, output

@calcfunction
def aiida_propose_occ_matrices_from_results(
    pk_list, N=8, debug=False, mode='random', *, self=None, tm_atoms=None, **kwargs):
    """
    AiiDA calcfunction that takes a list of PKs
    and returns a list of PKs of Dict nodes that are themselves stored
    and contain the occupation matrices. 

    This function wraps `propose_new_constraints` to create the Dict nodes.
    
    :param pk_list: List of PKs to load the occupation matrices from a AFMScanWorkChain or ConstrainedScanWorkChain.
    :param N: Int, number of dictionaries to return.
    :param debug: Bool, whether to print debug information.
    :param mode: Mode for selecting the dictionaries, e.g., 'random' or 'read'.
    :param kwargs: Additional keyword arguments for `propose_new_constraints`.

    :return: List of Dict nodes containing the occupation matrices.

    !!! WARNING PRINT STATEMENTS !!!

    This function uses print statements to log debug information.
    This is because it is a calcfunction wrapping AiiDA agnostic code.
    The print statements will be captured in the AiiDA report log.
    """

    # load the nodes from the PKs and convert to unified format
    occ_matrices = []
    for pk in pk_list.get_list():
        node = load_node(pk)
        
        # Handle JsonableData nodes containing OccupationMatrixData (preferred)
        if hasattr(node, 'obj') and hasattr(node.obj, 'as_dict'):
            # This is a JsonableData node containing our OccupationMatrixData
            occupation_matrix_data = node.obj
            occ_matrices.append(occupation_matrix_data)
            print(f"Loaded occupation matrix from JsonableData node with PK {pk}")
        
        # Legacy support for saved matrix as Dict
        elif node.__class__.__name__ == "Dict":
            legacy_dict = node.get_dict()
            occupation_matrix_data = OccupationMatrixData.from_legacy_dict(legacy_dict)
            occ_matrices.append(occupation_matrix_data)
            # print a deprecated warning
            print(f"Warning: Loaded occupation matrix from Dict node with PK {pk}. This is deprecated, please use OccupationMatrixAiidaData.")
        
        # Handle calculation nodes directly (for backward compatibility)
        elif hasattr(node, 'process_type') and ('aiida.calculations:quantumespresso.pw' in node.process_type or 'aiida.calculations:lordcapulet.constrained_pw' in node.process_type):
            # try to get the output occupation matrix from the CalcJobNode using unified extractor
            try:
                occupation_matrix_data = extract_occupations_from_calc(node)
                occ_matrices.append(occupation_matrix_data)
                print(f"Warning: Extracted occupation matrix directly from CalcJobNode with PK {pk}. Consider using workchain outputs instead.")
            except Exception as e:
                raise ValueError(f"CalcJobNode with PK {pk}, error in parsing occupation_matrix: {e}")
        
        else:
            raise ValueError(f"Unsupported node type for PK {pk}: {type(node)}. Expected OccupationMatrixAiidaData, Dict, or CalcJobNode.")
        

    # now get the N dictionaries from the list


    # Convert AiiDA data types to native Python types for the internal function call
    # This is necessary because propose_new_constraints expects standard Python types,
    # but AiiDA calcfunctions receive AiiDA node types (Dict, List, Int, Float, Str)
    kwargs_internal = {}
    for key, value in kwargs.items():
        # Handle AiiDA Dict and List nodes by extracting their content
        if isinstance(value, (Dict, List)):
            # For List nodes: get_list() returns the Python list
            # For Dict nodes: get_dict() returns the Python dictionary
            kwargs_internal[key] = value.get_list() if isinstance(value, List) else value.get_dict()
        # Handle AiiDA numeric and string nodes by extracting their .value attribute
        elif isinstance(value, (Int, Float, Str)):
            kwargs_internal[key] = value.value
        # Raise error for any unsupported AiiDA node types
        else:
            raise ValueError(f"Unsupported AiiDA node type for key '{key}': {type(value)}. "
                           f"Only Dict, List, Int, Float, and Str nodes are supported.")
    # check if this ran in the debug mode
    if debug and self is not None:
        self.report(f"Loaded {len(occ_matrices)} occupation matrices from nodes with PKs: {pk_list.get_list()}")
        self.report(f"Using proposal mode: {mode.value} with N = {N.value} samples per generation")



    # Filter atoms by species if tm_atoms is provided
    if tm_atoms is not None:
        tm_atoms_list = tm_atoms.get_list() if hasattr(tm_atoms, 'get_list') else tm_atoms
        filtered_matrices = []
        for occ_matrix_data in occ_matrices:
            filtered_data = filter_atoms_by_species(occ_matrix_data, tm_atoms_list)
            filtered_matrices.append(filtered_data)
        occ_matrices = filtered_matrices

    # TODO: REFACTOR NEEDED - This is a temporary workaround that loses metadata
    # Currently converting to legacy format for internal processing, which causes:
    # 1. Loss of 'specie' and 'shell' information from input occupation matrices
    # 2. Proposals end up with 'Unknown' specie and 'UNKNOWN' shell
    # 
    # SOLUTION: Refactor propose_new_constraints() and proposal_modes to work directly
    # with OccupationMatrixData instead of legacy dict format. This will preserve
    # all metadata (specie, shell) throughout the proposal generation pipeline.
    # The proposal functions should accept and return OccupationMatrixData objects.
    legacy_occ_matrices = []
    for occ_matrix_data in occ_matrices:
        legacy_dict = occ_matrix_data.to_legacy_dict()
        legacy_occ_matrices.append(legacy_dict)
    
    print(legacy_occ_matrices)


    # magic happens here
    proposals, to_report = redirect_print_report(
                                    propose_new_constraints,
                                    occ_matr_list=legacy_occ_matrices,
                                    N=N.value,
                                    debug=debug.value,
                                    mode=mode.value,
                                    **kwargs_internal
                                    )

    # this does not work as it should, needs refactoring

    if self is not None:
        self.report(to_report)

    # TODO: REFACTOR NEEDED - Metadata loss during proposal conversion
    # Currently proposals only contain matrix data without specie/shell information.
    # When converting back via from_constrained_matrix_format(), we get:
    #   - specie = 'Unknown' 
    #   - shell = 'UNKNOWN'
    # instead of preserving the original values from input occupation matrices.
    #
    # SOLUTION: Pass the original occ_matrices (OccupationMatrixData) along with proposals
    # to extract and preserve specie/shell metadata when creating output nodes.
    # Alternative: Refactor propose_new_constraints to work with OccupationMatrixData directly.
    dict_nodes = []
    for proposal in proposals:
        # Convert the proposal dict to OccupationMatrixData format
        # proposal has format {'matrix': [atom][spin][orbital][orbital]}
        # Need to convert to our unified format
        from lordcapulet.utils import OccupationMatrixData
        occ_data = OccupationMatrixData.from_constrained_matrix_format(proposal)
        
        # Store as JsonableData
        json_node = JsonableData(occ_data)
        json_node.store()
        dict_nodes.append(json_node)

    # return a list of the PKs of the Dict nodes
    return List(list=[node.pk for node in dict_nodes])


def propose_new_constraints(occ_matr_list, N, mode='random', debug=True, **kwargs):
    """
    !!IMPORTANT!! THIS FUNCTION SHOULD NOT GET ANY AIIDA TYPES AS INPUT
    
    Returns a list of N dictionaries from a list of dictionaries.
    
    This will be a giant function with a lot of logic
    and it is better that everything non trivial gets its own function and wrapped here

    
    :param occ_matr_list: List of dictionaries to choose from.
    :param N: Number of dictionaries to return.
    :return: List of N dictionaries.
    """
    # make sure that N is > 1

    atom_names = list(occ_matr_list[1].keys())
    natoms = len(atom_names)
    spin_names = ['up', 'down']
    nspin = 2 # up and down spin

    norbitals = np.array(occ_matr_list[1][atom_names[0]]['spin_data']['up']['occupation_matrix']).shape[0] 

    # structure of an output dictionary
    # { 'matrix': array of shape (natoms, nspin, norbitals, norbitals) }'
    if debug:
        print(f"Number of atoms: {natoms}")
        print(f"Number of spins: {nspin}")
        print(f"Number of orbitals for atom 1: {norbitals}")


    if N < 1:
        raise ValueError("N must be greater than or equal to 1")

    # implement case switch for mode
    match mode:

        case 'random':
            proposals = propose_random_constraints(occ_matr_list, natoms,  N, debug=debug, **kwargs)

        case 'random_so_n':
            proposals = propose_random_so_n_constraints(occ_matr_list, natoms, N, debug=debug, **kwargs)

        case 'read':
            # check if there is readfile in kwargs
            if 'readfile' not in kwargs:
                raise ValueError("readfile must be provided in kwargs for read mode")
            readfile = kwargs['readfile']
            # read the json file
            with open(readfile, 'r') as f:
                loaded_data = json.load(f)
            # assert that the loaded list is longer than N
            if len(loaded_data) < N:
                raise ValueError(f"Loaded data has only {len(loaded_data)} dictionaries, but N is {N}")

            # now loaded_data[iteration]['occupation_numbers'][iatom][ispin] is a norbital x norbital matrix


            #create now a list of dictionaries
            target_matrix_np = np.zeros((natoms, nspin, norbitals, norbitals))

            proposals = []
            for iteration in range(0, N):
                proposal = {}
                for iatom in range(natoms):
                    for ispin in range(nspin):
                        target_matrix_np[iatom, ispin] = \
                            np.array(loaded_data[iteration]['occupation_numbers'][iatom][ispin])
                proposal['matrix'] = target_matrix_np.tolist()
                proposals.append(proposal)
            if debug:
                print(f"Reading {N} dictionaries from file")

        
        
    
    return proposals