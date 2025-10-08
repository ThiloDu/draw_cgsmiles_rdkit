import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import cgsmiles

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


# Snippets for CG coordinates, currently not working
# def correct_position_assignments(res_graph):
#     '''
#     Correct the position assignments in the resolved graph. 
#     Needed because cgsmiles assigns positions in order, disregarding U nodes.
#     U nodes are set to the origin initially, then moved to the centroid of their connected nodes.

#     Parameters:
#     ----------
#     res_graph : networkx.Graph
#         The resolved graph with initial position assignments.

#     Returns:
#     -------
#     res_graph : networkx.Graph
#         The resolved graph with corrected position assignments.
#     '''
#     position_dict = {node_id: data['position'] for node_id, data in res_graph.nodes(data=True)}
#     offset = 0
#     for n, d in res_graph.nodes(data=True):
#         if d['fragname'] == 'U':
#             offset += 1
#             res_graph.nodes[n]['position'] = np.array([0.0, 0.0, 0.0])
#         else:
#             res_graph.nodes[n]['position'] = position_dict[n-offset]

#     for n in res_graph.nodes:
#         d = res_graph.nodes[n]
#         if d['fragname'] == 'U':
#             connected_nodes = [edge[1] if edge[0] == n else edge[0] for edge in res_graph.edges(n)]
#             if len(connected_nodes) == 0:
#                 raise ValueError(f'Warning: Unconnected U node at {n}, skipping')
#             coords = [res_graph.nodes[c]["position"] for c in connected_nodes if res_graph.nodes[c]["fragname"] != 'U']
#             d['position'] = np.mean(coords, axis=0)
#     return res_graph

# def write_file(file_path, lines) -> None:
#     '''
#     Function that writes a list of lines to a text file.
#     '''
#     # check if lines is a list of strings
#     if not all(isinstance(line, str) for line in lines):
#         raise ValueError('lines must be a list of strings')
#     with open(file_path, 'w') as file:
#         file.writelines(lines)
#     return 

# resname = 'A10'
# pdb_lines = []
# offset = 0
# for n, d in res_graph.nodes(data=True):
#     coords = d["position"]
#     line = f'HETATM{n+1:5d}  {d["fragname"][:3]:<3s} {resname:<3s}     1    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00            \n'
#     pdb_lines.append(line)
# pdb_lines.append('END\n')
# write_file(f'{resname}.pdb', pdb_lines)