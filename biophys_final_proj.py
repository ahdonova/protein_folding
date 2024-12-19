import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# Load the structure (native state) from a PDB file
u = mda.Universe("1ico.pdb")

# Select all residues in the structure
residues = u.residues

# Set a distance cutoff for contacts (in angstroms)
cutoff = 4.0  # distance defines a contact

# Initialize a contact matrix (residues x residues)
n_residues = len(residues)
contact_matrix = np.zeros((n_residues, n_residues))

# Loop over all pairs of residues
for i in range(n_residues):
    for j in range(i + 1, n_residues):  # avoid double-counting
        # Get the atoms of residues i and j
        atoms_i = residues[i].atoms
        atoms_j = residues[j].atoms

        # Loop over all pairs of atoms from residues i and j
        for atom_i in atoms_i:
            for atom_j in atoms_j:
                # Calculate the distance between atoms
                distance = np.linalg.norm(atom_i.position - atom_j.position)
                if distance < cutoff:
                    # If atoms are within the cutoff, mark the residues as in contact
                    contact_matrix[i, j] += 1
                    contact_matrix[j, i] += 1  # symmetric matrix

# Normalize the contact matrix (binary contact or no contact)
contact_matrix[contact_matrix > 0] = 1

# Create a contact map
plt.imshow(contact_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Contact (1 = in contact, 0 = not in contact)')
plt.title('Residue Contact Map (Native State)')
plt.xlabel('Residue Index')
plt.ylabel('Residue Index')


plt.tight_layout()

#calculating energy
import math

# Constants
dS = -.01 # Entropic penalty per folded residue, .5 * kb in coil, 1 kb elsewhere
T = 300.0  # Temperature in Kelvin (example: 300K)
kb = 1.380649 * 10**-23  # Boltzmann constant in J/K
N = len(residues)  # Number of residues

# generate random conformation matrix
conformation = np.random.choice([0, 1], size=(N))
native_conformation = [1]*N

# 1 if residues i,j contact in native state, 0 otherwise
def native_contact(i, j):
    if contact_matrix[i][j]:
        return 1
    else:
        return 0

# Returns the contact energy for the pair of residues i and j
def contact_energy(i, j):
    return -1  # Arbitrary negative value for now (stabilizing contact)

# Returns 1 when all residues between i and j are in the native state
def m(i, j, conformation):
    x = i
    while x <= j:
        if conformation[x] == 0:  # If any residue is not in native conformation, return 0
            return 0
        x += 1
    return 1  # If all residues between i and j are native, return 1

# Returns the Hamiltonian H for the given conformation
def find_H(conformation):
    H = 0.0
    i = 0
    while i < N:
        j = i + 1
        while j < N:
            H += native_contact(i, j) * contact_energy(i, j) * m(i, j, conformation)
            j += 1
        i += 1
    return H

# Returns the entropy S for the given conformation
def find_S(conformation):
    S = 0.0
    i = 0
    while i < N:
        S += conformation[i] * dS
        i += 1
    return S

# Returns the fraction of native residues in the conformation
def find_n(conformation):
    i = 0
    ret = 0
    while i < N:
        ret += conformation[i]
        i += 1
    return ret / N

import random

def generate_correlated_conformations(N, num_conformations, correlation_prob=0.7):
    conformations = []
    for _ in range(num_conformations):
        conformation = [random.choice([0, 1])]
        for _ in range(1, N):
            if random.random() < correlation_prob:
                conformation.append(conformation[-1])  # Extend same state
            else:
                conformation.append(1 - conformation[-1])  # Flip state
        conformations.append(conformation)
    return conformations



# Finds the partition function Z for a list of conformations
def find_Z(conformations_list):
    Z = 0.0
    for conformation in conformations_list:
        # Partition function calculation
        H = find_H(conformation)
        S = find_S(conformation)
        Z += math.exp((1 / (kb * T)) * (H - T * S))
    return Z

#plots free energy vs reaction progression (n)
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def store_and_plot_average_free_energy(conformations, T):
    """
    Calculate and store the average free energy for each n value, 
    and plot it against parameter n.

    Args:
        conformations (list): A list of protein conformations, 
                              where each conformation is a list of 0s and 1s.
        T (float): Temperature for the free energy calculation.
    """
    # Dictionary to store free energies for each n value
    free_energy_by_n = defaultdict(list)

    # Calculate free energy and group by n
    for conformation in conformations:
        H = find_H(conformation)  # Calculate Hamiltonian
        S = find_S(conformation)  # Calculate entropy
        n = find_n(conformation)  # Calculate the parameter n
        F = H - T * S  # Calculate free energy
        free_energy_by_n[n].append(F)  # Store free energy for the corresponding n

    # Calculate the average free energy for each n
    n_values = sorted(free_energy_by_n.keys())  # Sort the n values
    avg_F_values = [np.mean(free_energy_by_n[n]) for n in n_values]  # Average free energy for each n

    # Plot average Free Energy vs n
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, avg_F_values, marker='o', color='blue', label='Average Free Energy (F)')
    plt.xlabel("n")
    plt.ylabel("Average Free Energy (F)")
    plt.title("Average Free Energy vs Folding Progression")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
N = len(residues)  # Length of the protein
num_conformations = 10000  # Number of conformations to generate
random_conformations = generate_correlated_conformations(N, num_conformations)

# Call the function to store and plot average free energy
store_and_plot_average_free_energy(random_conformations, T)


    

'''
Hconf = find_H(conformation)
Sconf = find_S(conformation)
Hnat = find_H(native_conformation)
Snat = find_S(native_conformation)
print("F:" + str(Hconf - T * Sconf) + ", H:" + str(Hconf) + ", S:" + str(Sconf) + ", n:" + str(find_n(conformation)))
print("F:" + str(Hnat - T * Snat) + ", H:" + str(Hnat) + ", S:" + str(Snat) + ", n:" + str(find_n(native_conformation)))
'''

def visualize_conformation(conformation):
    """
    Visualizes protein residues in native (blue) and nonnative (yellow) states.

    Parameters:
        residues (list): A list of integers where 1 represents native and 0 represents nonnative.
    """
    # Set up colors
    colors = ['yellow' if state == 0 else 'blue' for state in conformation]

    # Create a bar chart
    plt.figure(figsize=(10, 2))
    bars = plt.bar(range(len(conformation)), [1] * len(conformation), color=colors, edgecolor='black')

    # Add labels and formatting
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("State", fontsize=12)
    plt.yticks([])  # Remove y-axis ticks for simplicity
    plt.title("Protein Residue States", fontsize=14)

    # Show the plot
    plt.tight_layout()
    
'''
visualize_conformation(conformation)
visualize_conformation(native_conformation)
plt.show()
'''