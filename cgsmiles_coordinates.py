import networkx as nx
from rdkit import Chem

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

import cgsmiles
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_AA_pdb(cgs_string, pdb_filename="molecule.pdb"):
    """
    Generate an all-atom PDB file from a CGSmiles string.

    Parameters
    ----------
    cgs_string : str
        The CGSmiles string representing the molecule.
    pdb_filename : str, optional
        The name of the output PDB file (default is "molecule.pdb").

    Returns
    -------
    None
        The function saves the PDB file to the specified filename.
    """
    _, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()

    mol, _ =  cgsmiles_to_rdkit(mol_graph, add_hydrogens=True)

    Chem.SanitizeMol(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    pdb_block = Chem.MolToPDBBlock(mol)
    with open(pdb_filename, "w") as f:
        f.write(pdb_block)

    print(f"PDB file saved as {pdb_filename}")