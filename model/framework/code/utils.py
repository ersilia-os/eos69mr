from math import ceil
from typing import Any

import click
from reinvent.models.model_factory.sample_batch import SampleBatch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import scaffoldgraph as sg


class SamplingResult:
    """A class to save the sampling result."""

    def __init__(self, input_items: "list[str]", output_items: "list[str]") -> None:
        self.input = input_items
        self.output = output_items

    def append(self, other: "SamplingResult"):
        self.input.extend(other.input)
        self.output.extend(other.output)
        pass


def are_smiles_same(smiles1: str, smiles2: str) -> bool:
    """To check whether given smiles are same."""
    # Parse SMILES strings to obtain RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Check if both SMILES strings are valid
    if mol1 is None or mol2 is None:
        return False  # Unable to parse one or both SMILES strings

    # Generate Morgan fingerprints for each molecule
    fp1 = Chem.AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)

    # Compare fingerprints to assess similarity
    try:
        similarity = float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except ValueError:
        similarity = -1
    # Determine whether the molecules are considered the same
    return similarity == 1.0


def filter_out_duplicate_molecules(sampled: SampleBatch, is_debug: bool) -> SampleBatch:
    """Filter out duplicate molecules from the sampled molecules.
    It also remove the output molecules if it is similar to input molecules.

    `sampled.items1` contains input smiles.
    `sampled.smilies` contains output smiles.
    """

    seen = {}
    items1 = []
    items2 = []
    states = []
    smilies = []
    nll_indices = []

    for item1, item2, smile, nll_index, state in zip(
        sampled.items1,
        sampled.items2,
        sampled.smilies,
        range(len(sampled.items1)),
        sampled.states,
    ):
        seen[item1] = item1

        if smile in seen:
            if is_debug:
                click.echo(
                    click.style(
                        f"Removing {smile}, as it is a duplicate entry.", fg="yellow"
                    )
                )
            continue

        seen[smile] = smile

        items1.append(item1)
        items2.append(item2)
        smilies.append(smile)
        states.append(state)
        nll_indices.append(nll_index)

    return SampleBatch(
        items1=items1,
        items2=items2,
        states=states,
        smilies=smilies,
        nlls=sampled.nlls[nll_indices],
    )


def pad_smiles(
    sampled: SamplingResult, input_smiles: "list[str]", target_length=100
) -> "list[str]":
    """For a given input smiles, it is not always possible to get the expected
    number (target_length) of output smiles (samples). This will cause
    problems in the downstream process. To mitigate this, the function
    pads the output with empty strings to match the target_length.
    """

    output_smiles: dict[str, list[str]] = {}

    # Initialize output_smiles with empty lists for each input smile
    output_smiles = {smile: [] for smile in input_smiles}

    # Smiles with similar tanimoto.
    similar_smiles = {}

    # Populate output_smiles with sampled smiles
    for idx, seq in enumerate(sampled.input):
        if seq in output_smiles:
            output_smiles[seq].append(sampled.output[idx])
        else:
            # REINVENT4 do some modification in the input smiles. Like if the
            # input smile is `CC(=O)Oc1ccccc1C(O)=O` then it will convert it
            # to `CC(=O)Oc1ccccc1C(=O)O`. Both of them are same, however both
            # strings are not same. This else condition will take care of this
            # edge case.
            if seq in similar_smiles:
                output_smiles[similar_smiles[seq]].append(sampled.output[idx])
            else:
                # Try to find similar smiles in input_smiles.
                for smile in input_smiles:
                    # smile and seq are in the following format: frag1|frag2.
                    # So we need to check whether frag1 and frag 2 of smile are
                    # equal to seq's frag1 and frag2

                    smile_frag1, smile_frag2 = smile.split("|")
                    seq_frag1, seq_frag2 = seq.split("|")

                    if are_smiles_same(smile_frag1, seq_frag1) and are_smiles_same(
                        smile_frag2, seq_frag2
                    ):
                        output_smiles[smile].append(sampled.output[idx])
                        similar_smiles[seq] = smile
                        break

    output = []

    # Construct the output list
    for smile in input_smiles:
        output_smile = output_smiles[smile][:target_length]
        output_smile_length = len(output_smile)
        if output_smile_length == target_length:
            output.extend(output_smile)
        else:
            output.extend(output_smile)
            padding = [""] * (target_length - output_smile_length)
            output.extend(padding)

    return output


def make_list_into_lists_of_n(lst: "list[str]", n: int) -> "list[list[str]]":
    """This function splits a list into n parts of equal size."""

    length = len(lst)

    if length % n != 0:
        # If length is not divisible by n, then
        # We don't have expected output.
        # Hence throwing error.
        raise Exception(f"{length} is not divisible by {n}")

    part_length = length // n

    output = []

    for start in range(0, length, part_length):
        end = start + part_length
        output.append(lst[start:end])

    return output


def get_smiles(mol):
    return Chem.MolToSmiles(mol)


def get_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def get_scaffold(mol):
    return MurckoScaffold.GetScaffoldForMol(mol)


def get_idxs_of_carbon_for_new_bond(mol):
    """Return indices of all carbon atoms available for new bond.

    Technically, it returns the indices of Carbon atoms that have
    at least one bond with hydrogen atom.
    """

    carbon_indices = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            bonds = atom.GetBonds()
            num_bonds = sum([b.GetBondTypeAsDouble() for b in bonds])
            if num_bonds < 4:
                carbon_indices.append(atom.GetIdx())

    return carbon_indices


def get_scaffold_and_attachment_points(mol):
    scaffold = get_scaffold(mol)

    if Chem.MolToSmiles(scaffold) == "":
        # If we have a valid scaffold then we are using it.
        scaffold = Chem.Mol(mol)

    return (scaffold, get_idxs_of_carbon_for_new_bond(scaffold))


def get_mol_after_adding_attachment_points_at(mol, at):
    connecting_atom = Chem.Atom("*")
    mutable_copy = Chem.RWMol(mol)

    for attachment_idx in at:
        connection_idx = mutable_copy.AddAtom(connecting_atom)
        mutable_copy.AddBond(attachment_idx, connection_idx, Chem.BondType.SINGLE)

    _mol = mutable_copy.GetMol()
    AllChem.Compute2DCoords(_mol)

    return _mol


def add_attachment_point(mol: "Any"):
    scaffold, at = get_scaffold_and_attachment_points(mol)
    return get_mol_after_adding_attachment_points_at(scaffold, at[:1])


def filter_duplicate_molecules(molecules):
    _mols = [get_smiles(mol) for mol in molecules]
    seen = {}
    filtered_list = []

    for m in _mols:
        if m in seen:
            continue
        seen[m] = m
        filtered_list.append(get_mol(m))

    return filtered_list


def can_frag_exists(base_mol, frag_1, frag_2):
    """
    A function that checks whether two fragments can exist without any overlapping in a base molecule.

    base_mol: The parent molecule (Mol).
    frag_1: First fragment (Mol).
    frag_2: Second fragment (Mol).
    """

    # Get positions of atoms of all substructures as sets of indices
    frag_1_sub_positions = [
        set(match) for match in base_mol.GetSubstructMatches(frag_1)
    ]
    frag_2_sub_positions = [
        set(match) for match in base_mol.GetSubstructMatches(frag_2)
    ]

    # Check for non-overlapping positions
    for position_1 in frag_1_sub_positions:
        for position_2 in frag_2_sub_positions:
            if position_1.isdisjoint(position_2):
                # Returns True if two sets don't have any common indexes between them
                return True

    return False


def get_unique_pairs_of_fragments(mol, fragments):
    """
    Generates all unique pairs of fragments and checks if they can coexist in the base molecule without overlapping.

    Parameters:
    mol: The base molecule (Mol).
    fragments: List of fragment molecules (list of Mols).

    Returns:
    List of tuples, where each tuple contains a pair of non-overlapping fragments.
    """
    pairs = []
    total_frags = len(fragments)

    for idx in range(total_frags):
        for j in range(idx + 1, total_frags):
            frag_1 = fragments[idx]
            frag_2 = fragments[j]

            if can_frag_exists(mol, frag_1, frag_2):
                pairs.append((frag_1, frag_2))

    return pairs


def get_largest_fragment(fragments):
    largest = fragments[0]
    wt = 0

    for fragment in fragments:
        w = Chem.Descriptors.ExactMolWt(fragment)

        if w > wt:
            wt = w
            largest = fragment

    return largest


def atom_remover(mol, pattern):
    """
    Returns the residue after removing the pattern from the molecule.
    """
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        yield Chem.Mol(mol)
    for match in matches:
        res = Chem.RWMol(mol)
        res.BeginBatchEdit()
        for aid in match:
            res.RemoveAtom(aid)
        res.CommitBatchEdit()

        yield res


def get_largest_scaffold_and_residue(mol, frags) -> "tuple[int, tuple[Any, Any]]":
    """
    Returns the largest scaffold/fragment and the largest residue after removing
    the largest scaffold/frament from the parent molecule.
    """
    largest = get_largest_fragment(frags)
    remaining = [x for x in atom_remover(mol, largest)][0]
    remaining = [Chem.MolFromSmiles(m) for m in Chem.MolToSmiles(remaining).split(".")]
    largest_remaining = get_largest_fragment(remaining)
    wt = Chem.Descriptors.ExactMolWt(largest) + Chem.Descriptors.ExactMolWt(
        largest_remaining
    )
    return (wt, (largest, largest_remaining))


def pick_from_unique_pairs(uq) -> "tuple[int, tuple[Any, Any]]":
    picked = (0, uq[0])

    for pairs in uq:
        mol_wt = Chem.Descriptors.ExactMolWt(pairs[0]) + Chem.Descriptors.ExactMolWt(
            pairs[1]
        )

        if picked[0] < mol_wt:
            picked = (mol_wt, pairs)

    return picked


def annotate_unannotated_smiles(smile: str) -> str:
    """Annotate the unannotated smiles."""
    mol = Chem.MolFromSmiles(smile)
    fragments = sg.get_all_murcko_fragments(mol, break_fused_rings=True)
    unique_pairs = get_unique_pairs_of_fragments(mol, fragments)

    picked = ()

    if len(unique_pairs) == 0:
        """If we are unable to find any unique pair of fragments, then we will pick the largest
        scaffold and the largest residue after removing the largest scaffold from the parent
        molecule.
        """
        picked = get_largest_scaffold_and_residue(mol, fragments)
    else:
        picked = pick_from_unique_pairs(unique_pairs)

    warhead_1 = add_attachment_point(picked[1][0])
    warhead_2 = add_attachment_point(picked[1][1])

    return f"{get_smiles(warhead_1)}|{get_smiles(warhead_2)}"
