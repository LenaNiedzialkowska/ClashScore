from Bio import PDB
import numpy as np
import re

# Inicjalizacja parsera PDB
pdb_parser = PDB.PDBParser(QUIET=True)

# Promienie atomow
atom_radius = {
    'H': 1.2,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
    'F': 1.47,
    'P': 1.8,
    'CL': 1.75,
    'MG': 1.73
}

# Zliczanie atomow
def count_atoms(structure):
    number_of_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    number_of_atoms.append(atom)
    return len(number_of_atoms)

def count_clashes(structure, clash_cutoff):
    """
    Liczy ilość nachodzeń między atomami w strukturze białka.

    :param structure: Struktura białka wczytana z pliku PDB.
    :param clash_cutoff: Próg nachodzenia, poniżej którego uznaje się, że atomy się nie nachodzą.
    :return: Liczba nachodzeń między atomami, podzielona przez 2 (każde nachodzenie jest zliczane dwukrotnie).
    """
    # Ustalenie, co traktujemy jako nachodzenie dla każdej pary atomów
    clash_cutoffs = {f"{i}_{j}": (atom_radius[i] + atom_radius[j] - clash_cutoff) for i in atom_radius for j in atom_radius}

    # Atomy dla których są promienie
    atoms = [atom for atom in structure.get_atoms() if atom.element in atom_radius]
    coords = np.array([a.coord for a in atoms], dtype="d")

    # KDTree do szybkiego wyszukiwania najbliższych sąsiadów
    kdtree = PDB.kdtrees.KDTree(coords)

    clashes = []

    # Iteracja przez wszystkie atomy
    for atom_1 in atoms:
        # Znalezienie atomów, które mogą się nachodzić
        kdtree_search = kdtree.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))

        # Pobranie indeksu i odległości potencjalnych nachodzeń
        potential_clashes = [(a.index, a.radius) for a in kdtree_search]

        # Iteracja przez potencjalne nachodzenia
        for ix, distance in potential_clashes:
            atom_2 = atoms[ix]

            # Wykluczenie nachodzeń od atomów w tym samym resztku lub sąsiednich resztkach
            if atom_1.parent.id == atom_2.parent.id or \
               atom_1.parent.id[1] == atom_2.parent.id[1] + 1 or \
               atom_1.parent.id[1] == atom_2.parent.id[1] - 1:
                continue

            # Obliczenie odległości między atomami
            x1, y1, z1 = atom_1.get_coord()
            x2, y2, z2 = atom_2.get_coord()
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

            # Obliczenie nachodzenia
            overlap = (atom_radius[atom_1.element] + atom_radius[atom_2.element] - distance)

            # Dodanie nachodzenia do listy
            if overlap >= 0.4:
                clashes.append((atom_1, atom_2))

    # Zwrócenie liczby nachodzeń podzielonej przez 2
    return len(clashes) // 2

# Ścieżka do pliku PDB
file_path = "R1107_reference.pdb"

# Wczytanie struktury z pliku PDB
protein_structure = pdb_parser.get_structure(file_path, file_path)

# Liczenie nachodzeń i liczby atomów
number_of_atoms = count_atoms(protein_structure)
num_clashes = count_clashes(protein_structure, 0.4)
clash_score = 1000 * num_clashes / number_of_atoms

# Wyświetlenie wyników
print("Number of atoms: ", number_of_atoms)
print("Number of clashes: ", num_clashes)
print("Clash score: ", clash_score)