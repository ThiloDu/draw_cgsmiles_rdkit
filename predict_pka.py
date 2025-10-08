import cgsmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from pkasolver.query import draw_pka_map, draw_pka_reactions, calculate_microstate_pka_values
import os, warnings, logging

def cgsmiles_to_rdkit(mol_graph, add_hydrogens=False):
    '''
    Convert a CGSmiles molecule graph to an RDKit molecule. 
    Automatically detects bond orders and aromaticity.
    Hydrogens will be skipped and restored later if needed.

    Parameters
    ----------
    mol_graph : networkx.Graph
        The molecular graph to convert. Needs to be atomistic resolution.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        The corresponding RDKit molecule.
    '''
    mol = Chem.RWMol()

    node_to_rdkit_idx = {}
    for node_idx, data in sorted(mol_graph.nodes(data=True)):
        if not add_hydrogens and data['element'] == 'H' :
            continue 
        atom = Chem.Atom(data['element'])
        atom.SetFormalCharge(data['charge'])
        rdkit_idx = mol.AddAtom(atom)
        node_to_rdkit_idx[node_idx] = rdkit_idx

    for i, j, data in mol_graph.edges(data=True):
        if not add_hydrogens and (mol_graph.nodes[i]['element'] == 'H' or mol_graph.nodes[j]['element'] == 'H'):
            continue  
        i = node_to_rdkit_idx[i]
        j = node_to_rdkit_idx[j]
        if data['order']==1:
            bond_order = Chem.rdchem.BondType.SINGLE
        elif data['order']==2:
            bond_order = Chem.rdchem.BondType.DOUBLE
        elif data['order']==3:
            bond_order = Chem.rdchem.BondType.TRIPLE
        elif data['order']==1.5:
            bond_order = Chem.rdchem.BondType.AROMATIC
        else:  
            raise ValueError(f"Unsupported bond order: {data['order']}") 
        mol.AddBond(i,j, bond_order)

    # add isomerism information if present
    isomer_data = [d[0] for d in nx.get_node_attributes(mol_graph, 'ez_isomer').values() if d is not None]

    for isomer in isomer_data:  
        isomer_type = isomer[4]
        isomer = [node_to_rdkit_idx[idx] for idx in isomer[:4]]  # convert to rdkit indices
        if isomer[1] > isomer[2]:
            continue
        bond = mol.GetBondBetweenAtoms(isomer[1], isomer[2])
        if not bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            print(f'WARNING: Isomerisation defined on non-double bond! (atoms {isomer[1]}-{isomer[2]})')
            print('    Check your CGSMILES string!')
            continue
        bond.SetStereoAtoms(isomer[0], isomer[3])
        if isomer_type == 'cis':
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)  # cis
        elif isomer_type == 'trans':
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
        else:
            raise ValueError(f"Unsupported isomer type: {isomer[4]}")
    return mol, node_to_rdkit_idx

class suppress_output:
    def __enter__(self):
        # open devnull
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        # save real stdout/stderr file descriptors
        self._save_stdout = os.dup(1)
        self._save_stderr = os.dup(2)
        # replace them with devnull
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        # restore stdout/stderr
        os.dup2(self._save_stdout, 1)
        os.dup2(self._save_stderr, 2)
        os.close(self._null_fd)
        os.close(self._save_stdout)
        os.close(self._save_stderr)


def predict_pka(cgs_string, show_protontation_reactions=False, show_protonation_map=False, print_pka_values=True):
    """
    Predict the pKa values of a molecule given its CGSmiles string.

    Parameters
    ----------
    cgs_string : str
        The CGSmiles string representing the molecule.

    Returns
    -------
    dict
        A dictionary with predicted pKa values and associated information.
    """

    # Resolve CGSmiles to molecular graph
    res_graph, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()
    
    # Convert molecular graph to RDKit molecule
    mol, cgs_to_mol_idx = cgsmiles_to_rdkit(mol_graph, add_hydrogens=True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.disable(logging.CRITICAL)

    with suppress_output():
        pka_results = calculate_microstate_pka_values(mol)
    
    IPythonConsole.molSize = 400,400
    if show_protontation_reactions:
        print('Drawing pKa reaction network...')
        display(draw_pka_reactions(pka_results))

    if show_protonation_map:
        print('Drawing pKa map...')
        display(draw_pka_map(pka_results))

    if print_pka_values:
        for state in pka_results:
            print(f'pKa = {state.pka}')
    return pka_results