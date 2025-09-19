import cgsmiles
from cgsmiles.drawing import draw_molecule
import pysmiles

import io
from PIL import Image
import cairosvg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D
from rdkit.Chem import rdDepictor,rdMolTransforms


def cgsmiles_to_rdkit(mol_graph):
    '''
    Convert a CGSmiles molecule graph to an RDKit molecule. 
    Automatically detects bond orders and aromaticity.

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

    for _, data in sorted(mol_graph.nodes(data=True)):
        mol.AddAtom(Chem.Atom(data['element']))

    for i, j, data in mol_graph.edges(data=True):
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
        if isomer[1] > isomer[2]:
            continue
        bond = mol.GetBondBetweenAtoms(isomer[1], isomer[2])
        if not bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            print(f'WARNING: Isomerisation defined on non-double bond! (atoms {isomer[1]}-{isomer[2]})')
            print('    Check your CGSMILES string!')
            continue
        bond.SetStereoAtoms(isomer[0], isomer[3])
        if isomer[4] == 'cis':
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)  # cis
        elif isomer[4] == 'trans':
            bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
        else:
            raise ValueError(f"Unsupported isomer type: {isomer[4]}")
    return mol

def _rotate_molecule_around_center(mol, theta):
    """
    Rotate a molecule in the XY plane around its centroid.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule to rotate. Must have 3D conformer.
    theta : float
        The rotation angle in degrees.

    Returns
    -------
    None
        The molecule is modified in place.
    """
    conf = mol.GetConformer(0)

    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = coords.mean(axis=0)

    # translation to origin
    T1 = np.eye(4)
    T1[:3, 3] = -centroid

    # rotation in xy plane around origin 
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c,  s, 0, 0],
        [-s, c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

    # translate centroid back to original position
    T2 = np.eye(4)
    T2[:3, 3] = centroid

    # Full transform = T2 * R * T1
    M = T2 @ R @ T1

    rdMolTransforms.TransformConformer(conf, M)

def _bead_radius(bead_type):
    '''
    Determine radius for the bead circle based on bead size given in the bead type.

    Parameters
    ----------
    bead_type : str
    
    Returns
    -------
    float
        The radius of the bead circle.
    '''
    match bead_type[0]:
        case 'U': # use small circles for dummy beads
            bead_radius = 7.5
        case 'T':
            bead_radius = 15
        case 'S':
            bead_radius = 20
        case 'R':
            bead_radius = 23.5
        case _:  # default to 'R' if size is not explicitly given
            bead_radius = 23.5 
    return float(bead_radius)

def _bead_color(bead_type):
    '''
    Determine color for the bead circle based on bead type.

    Parameters
    ----------
    bead_type : str

    Returns
    -------
    str
        The color of the bead circle in hex format.
    '''
    if bead_type[0] in ['T', 'S', 'R']: # remove size designation if present
        bead_type = bead_type[1]
    else:
        bead_type = bead_type[0]
        
    match bead_type:
        case 'C':
            color = '#cccccc' # light gray
        case 'N':
            color = '#80b3ff' # blue
        case 'P':
            color = '#ff0000' # red
        case 'X':
            color = '#ccff00' # lime
        case 'Q':
            color = '#ffcc00' # yellow
        case 'D':
            color = '#ff8000' # orange
        case 'U':
            color = '#8e008e' # purple
        case _:
            color = '#808080' # gray
    return color

def _create_bead_circle(bead_position, name):
    '''
    Create SVG circle element for a bead.
    
    Parameters
    ----------
    bead_position : list or np.array
        The (x, y) position of the bead.
    name : str
        The bead type/name.
    
    Returns
    -------
    str
        The SVG circle element as a string.
    '''
    circle_svg = ['<circle\n']
    circle_svg += f'\tstyle="opacity:0.5;fill:{_bead_color(name)};fill-opacity:1;stroke:#6e6e6e;stroke-width:1.75;stroke-miterlimit:4;stroke-dasharray:none"\n'
    circle_svg += f'\tid="circle_bead_{name}"\n'
    circle_svg += f'\tcx="{bead_position[0]}"\n'
    circle_svg += f'\tcy="{bead_position[1]}"\n'
    circle_svg += f'\tr="{_bead_radius(name)}" />\n'
    return ''.join(circle_svg)

def _create_bead_label(bead_position, bead_type, bead_name=None, type_color='#666666', name_color='#000000'):
    '''
    Create SVG text element for a bead label.

    Parameters
    ----------
    bead_position : list or np.array
        The (x, y) position of the center of the bead.
    bead_type : str
        The bead type/name.
    bead_name : str, optional, default: None
        The bead name to display above the bead type. If None, only the bead type is shown.
    type_color : str, optional, default: '#666666'
        The color of the bead type text in hex format.
    name_color : str, optional, default: '#000000'
        The color of the bead name text in hex format.   
    
    Returns
    -------
    str
        The SVG text element as a string.
    '''
    fontsize = 13
    position = [bead_position[0] + 1/np.sqrt(2) * _bead_radius(bead_type), bead_position[1] - 1/np.sqrt(2) *_bead_radius(bead_type)] 

    # define text element
    text_svg = ['<text\n']
    text_svg += f'\tid="text_bead_{bead_type}"\n'
    text_svg += f'\tstyle="font-size:{fontsize}px;dominant-baseline:bottom;text-anchor:start;font-family:sans-serif;"\n'
    text_svg += f'\tx="{position[0]}"\n'
    text_svg += f'\ty="{position[1]}">\n'

    if bead_type == 'U' and bead_name is not None: # only show bead name in black
        type_color = '#000000' # use black color for virtual nodes
        bead_type = bead_name
    
    # define tspan element for bead name if given
    elif bead_name is not None:
        text_svg += f'\t<tspan x="{position[0]}" y="{position[1]-fontsize}" font-size="{fontsize}px" fill="{name_color}" font-family="sans-serif">{bead_name}</tspan>\n'
    # define tspan element for bead type
    text_svg += f'\t<tspan x="{position[0]}" y="{position[1]}" font-size="{fontsize}px" fill="{type_color}" font-family="sans-serif">{bead_type}</tspan>\n'
    text_svg += '</text>\n'
    return ''.join(text_svg)

def detect_virtual_nodes(resgraph):
    """
    Detect virtual edges (nodes only connected by edges with 'order' == 0)

    Parameters
    resgraph : networkx.Graph
        The residue graph to analyze.

    Returns
    dict
        A dictionary where keys are the index of the virtual nodes,
        and values are sorted lists of the indices of the nodes they are connected to.
    """
    virtual_edges = {}
    for node in resgraph.nodes:
        edges = resgraph.edges(node, data=True)
        # If node has no edges, skip
        if not list(edges):
            continue
        # Check if all edges have 'order' == 0
        if all(edge_data.get('order', None) == 0 for _, _, edge_data in edges):
            # save the indices of the edges
            virtual_edges[node] = sorted([v if u == node else u for u, v, _ in edges])
    return virtual_edges

def correct_graph_assignment(resgraph):
    '''
    Correct the graph assignment in the residue graph if virtual nodes are present.
    Currently CGSmiles assigns the atom graph of the next node to the virtual node, shifting all subsequent assignments by one.
    This is a workaround to fix that issue. Might be removed in future versions of CGSmiles.

    Parameters
    ----------
    resgraph : networkx.Graph
        The residue graph to correct.

    Returns
    -------
    networkx.Graph
        The corrected residue graph.
    '''
    virtual_edges = detect_virtual_nodes(resgraph)

    for node in resgraph.nodes:
        if node in virtual_edges.keys():
            for i in range(node, len(resgraph.nodes)-1)[::-1]:
                # shift graph one position up to compensate the wrong graph assignment done by cgsmiles
                resgraph.nodes(data=True)[i+1]['graph'] = resgraph.nodes(data=True)[i]['graph']
            resgraph.nodes(data=True)[node]['graph'] = nx.Graph() # set empty graph for virtual node
    return resgraph

def draw_beads(svg, res_graph, full_mol, drawer_coords, include_hydrogens, show_vs, add_names, show_node_indicators):
    '''
    Draw beads on top of the rdkit molecule SVG.
    Bead positions are calculated as the center of geometry of their constituent atoms.
    Virtual nodes (connected only by edges with 'order' == 0) are placed at the center of geometry of their connected beads.    

    Parameters
    ----------
    svg : str
        The SVG string of the molecule drawing.
    res_graph : networkx.Graph
        The residue graph containing bead information.
    full_mol : rdkit.Chem.rdchem.Mol
        The full RDKit molecule with all atoms including hydrogens.
    drawer_coords : rdMolDraw2D.MolDraw2DSVG
        An RDKit drawer object used to get atom coordinates.
    include_hydrogens : bool
        Whether hydrogens are to be included in the bead position calculation.
    show_vs : bool
        Whether to show virtual nodes.
    add_name : bool, optional
        Whether to add bead names next to the beads, by default False.
    show_node_indicators : bool, optional
        Whether to show trailing capital letter from bead type if present.

    Returns
    -------
    str
        The modified SVG string with beads drawn on top.
    '''
    # Draw full molecule to get atom coordinates
    drawer_coords.DrawMolecule(full_mol)
    drawer_coords.FinishDrawing()  # optional but safe

    correct_graph_assignment(res_graph) # ensure correct mapping if virtual nodes are present
    virtual_edges = detect_virtual_nodes(res_graph)

    bead_positions = {} 
    for bead, bead_data in res_graph.nodes(data=True):
        if bead in virtual_edges.keys(): # skip virtual nodes for now, will be added later
            continue

        bead_type = bead_data['fragname']
        bead_name = bead_data.get('n', None)

        # compute bead position as center of geometry of its atoms
        coords = []
        for atoms, data in bead_data['graph'].nodes(data=True):
            if not include_hydrogens and data['element'] == 'H':
                continue
            coords.append(drawer_coords.GetDrawCoords(atoms))
        bead_position = np.mean(coords, axis=0)
        bead_positions[bead] = bead_position
        
        # draw bead as circle and optionally add bead name
        circle_svg = _create_bead_circle(bead_position, bead_type)
        svg = svg.replace('</svg>', circle_svg + '</svg>')
        if add_names: # place bead type top right of bead
            if not show_node_indicators: # remove trailing capital letter from bead type if present
                bead_type = bead_type[:-1] if bead_type[-1].isupper() and len(bead_type) > 1 else bead_type
            text_svg = _create_bead_label(bead_position, bead_type, bead_name)
            svg = svg.replace('</svg>', text_svg + '</svg>')

    if show_vs: # add virtual nodes if requested
        for virtual_node, connections in virtual_edges.items():
            bead_type = res_graph.nodes[virtual_node]['fragname']
            bead_name = res_graph.nodes[virtual_node].get('n', None)
            # compute bead position as center of geometry of its connected beads
            coords = []
            for bead in connections:
                coords.append(bead_positions[bead])
            bead_position = np.mean(coords, axis=0)
            bead_positions[virtual_node] = bead_position  
            
            # draw bead as circle and optionally add bead name
            circle_svg = _create_bead_circle(bead_position, bead_type)
            svg = svg.replace('</svg>', circle_svg + '</svg>')
            if add_names: # place bead type top right of bead
                text_svg = _create_bead_label(bead_position, bead_type, bead_name) # use black color for virtual nodes
                svg = svg.replace('</svg>', text_svg + '</svg>')
    return svg

def draw_mapping(cgs_string, name=None, show_hydrogens=False, include_hydrogen_in_bead_position=None, show_mapping=True, show_bead_labels=False, show_vs=True, show_atom_indices=False, color_atoms=False, show_image=True, show_node_indicators=False, rotate_by=0, canvas_size=(300,300), scale_factor=25):
    '''
    Draw a CGSmiles molecule with optional bead mapping overlay using RDKIT.
    Molecules are not scaled to fit the canvas, in order to keep relative sizes of beads and atoms.
    For large molecules, increase canvas_size or change scale factor.
    The bead positions are calculated as the center of geometry of their constituent atoms.
    Virtual nodes (connected only by edges with 'order' == 0) are placed at the center of geometry of their connected beads.
    Bead names are added as labels. They can be added via a 'n' annotation in the CGSmiles string.

    Example usage:
    ```
    draw_mapping('{[#TC5]1.2[#TC5].3[#TC5]1.([#U]23)}.{#TC5=[$]cc[$][>]}', 'benzene')
    ```

    Parameters
    ----------
    cgs_string : str
        The CGSmiles string to visualize.
    name : str, optional
        The name to save the SVG file as (without extension). If None (default), the SVG is not saved.
    show_hydrogens : bool, optional, default: False
        Whether to show hydrogens in the molecule drawing. Default is False.
    include_hydrogen_in_bead_position : bool, optional, default: None
        Whether to include hydrogens in the bead position calculation. If None (default), this is set to the value of show_hydrogens.
    show_mapping : bool, optional, default: True
        Whether to overlay the bead mapping on the molecule.
    show_bead_labels : bool, optional, default: False
        Whether to show bead type labels next to the beads.
    show_vs : bool, optional, default: True
        Whether to show virtual nodes (connected only by edges with 'order' == 0).
    show_atom_indices : bool, optional, default: False
        Whether to show atom indices in the molecule drawing.
    color_atoms : bool, optional, default: False
        Whether to color atoms by element. 
    show_image : bool, optional, default: True
        Whether to display the image using matplotlib.
    show_node_indicators : bool, optional, default: False
        Whether to show trailing capital letter from bead type if present. Used to differenciate different mappings of the same bead type.
    rotate_by : float, optional, default: 0
        Rotate the molecule by this angle (in degrees) in the XY plane around its centroi
    canvas_size : tuple, optional, default: (300, 300)
        The size of the drawing canvas in pixels. 
    scale_factor : int, optional, default: 25
        The scale factor for drawing. Higher values result in larger drawings. 

    Returns
    -------
    None
    '''
    
    if not name and not show_image:
        raise ValueError("Either name must be provided to save the SVG or show_image must be True to display the image.")
    if not show_mapping and show_bead_labels:
        print("Warning: show_bead_labels is set to True but show_mapping is False. Bead labels will not be displayed.")
    if include_hydrogen_in_bead_position is None: # if not explicitly set, use show_hydrogens as indicator
        include_hydrogen_in_bead_position = show_hydrogens
    def setup_drawer():
        W, H = canvas_size # default 300 x 300
        minv = Point2D(1000, 1000)
        maxv = Point2D(-1000, -1000)
        scalex, scaley = [scale_factor, scale_factor] # scale factor for drawing
        drawer = rdMolDraw2D.MolDraw2DSVG(W, H)
        drawer.SetScale(scalex, scaley, minv, maxv)
        return drawer

    # Prepare molecule 
    res_graph, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()
    full_mol = cgsmiles_to_rdkit(mol_graph)
    rdDepictor.Compute2DCoords(full_mol) # ensure 2D coordinates are present for the full molecule 
    _rotate_molecule_around_center(full_mol, rotate_by)

    if not show_hydrogens:
        mol = Chem.RemoveHs(full_mol, updateExplicitCount=True, sanitize=False) # remove hydrogens for final drawing but keep the original coordinates  
    else:
        mol = full_mol

    # Prepare drawer for final drawing
    drawer_final = setup_drawer()
    drawer_final.drawOptions().addAtomIndices = show_atom_indices
    drawer_final.drawOptions().bondLineWidth = 1.5
    if not color_atoms: # disable coloring by element
        drawer_final.drawOptions().useBWAtomPalette()
    drawer_final.DrawMolecule(mol)
    drawer_final.FinishDrawing()
    svg = drawer_final.GetDrawingText()

    if show_mapping:
        svg = draw_beads(svg, res_graph, full_mol, setup_drawer(), include_hydrogen_in_bead_position, show_vs, show_bead_labels, show_node_indicators)

    if name: # save SVG if name is given
        with open(f"{name}.svg", "w") as f:
            f.write(svg)
    
    if show_image: # display image using matplotlib
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        img = Image.open(io.BytesIO(png))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def draw_mapping_default(cgs_string:str, show_hydrogens = False, ax=None):
    '''
    Draw a CGSmiles molecule with default settings using matplotlib.

    Parameters
    ----------
    cgs_string : str
        The CGSmiles string to visualize.
    show_hydrogens : bool, optional, default: False
        Whether to show hydrogens in the molecule drawing. Default is False.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes to draw on. If None (default), a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the drawn molecule.
    '''
    _, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()

    if ax is None:
        _, ax = plt.subplots(1, 1)
    if not show_hydrogens:
        pysmiles.remove_explicit_hydrogens(mol_graph)
        labels = {}
        for node in mol_graph.nodes:
            hcount =  mol_graph.nodes[node].get('hcount', 0)
            label = mol_graph.nodes[node].get('element', '*')
            if hcount > 0:
                label = label + f"H{hcount}"
            labels[node] = label
        draw_molecule(mol_graph, ax=ax, labels=labels, scale=1, align_with='x')
    else:
        draw_molecule(mol_graph, ax=ax, scale=1)
    ax.set_frame_on('True')
    return ax