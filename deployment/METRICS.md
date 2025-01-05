## User Guidelines

Welcome to the Control-GeoLDM model interface. Below are some guidelines to help you get started:

### 1. **Upload Protein Pocket Files**
- You can upload multiple protein pocket PDB files at once.

</br>

### 2. **Download Results**
- After generation is done, you may download the generated ligand SDF files via the provided zip file. If Docking is performed, the docked ligand and pocket PDBQT files will also be available.

</br>

### 3. **View Metrics**
- Multiple metrics will also be computed for the generated ligands. You may view them in the <code>Metrics</code> tab. They will also be available in the results zip file.

---

</br>

## Model Configurations

### Model Selection
- You may select the model to use from the dropdown section. 
- Models that contain <code>CA</code> in their names are models that will extract the Alpha Carbon atom of each residue in the protein pockets and use them as representation of the pocket during generation. 
- Models that DO NOT contain <code>CA</code> in their names will use the full atom representation of the pocket during generation.

</br>

### Random Seed
- Random seed sets the initial seed value for all randomizers used throughout the program other than Docking Analysis. 
- Different seeds can result in different variations of the generated ligands, even if the same model and protein pocket files are used. 
- Random seed is strictly a positive integer no larger than <code>4,294,967,295</code>.

</br>

### Batch Size
- Batch Size controls the number of ligand generations that the model will execute generally. A larger batch size allows for faster computation and has zero effect on the generated ligands. 
- The maximum batch size allowed for each model differs and depends on the size of the available GPU.

---

</br>

## Ligand Generation Configurations

### Number of Ligand Samples to Generate per Pocket
- This sets the number of ligand samples to generate for each provided protein pocket.

</br>

### Randomly Sample Number of Atoms per Ligand from Pre-computed Distribution
- When enabled (set to True), the number of atoms in each generated ligand (not inclusive of Hydrogen atoms) will be sampled from a distribution that was pre-computed based on the model's training dataset. 
- This distribution provides insights into the typical atom count observed during training, and is visualized as a plot for better understanding.

</br>

### Delta Number of Atoms per Ligand
- When the random sampling of number of atoms is activated (as described above), the Delta Number of Atoms per Ligand allows you to modify the sampled atom count. 
- Specifically, this parameter gives you the flexibility to fine-tune the atom count for each generated ligand by applying an increase to the sampled value. The final number of atoms is given by:

$$N_{final} = N_{sampled} + N_{delta}$$
​

### Number of Atoms per Ligand
- This option is only available when the random sampling of number of atoms is deactivated. 
- This will allow you to manually set the number of atoms for each generated ligand (not inclusive of Hydrogen atoms).

---

</br>

## Docking Analysis Configurations

Docking Analysis will be performed using [QuickVina2.1](https://github.com/QVina/qvina/raw/master/bin/)

</br>

### Molecule Fragment Size
- This threshold filters and retains generated ligands that have the largest fully connected fragments equal to or above this connectivity threshold for docking analysis.

</br>

### Search Space XYZ Dimensions (Angstroms)
- Dimensions of the docking search space (cuboid) in Angstroms.

</br>

### Search Exhaustiveness
- Exhaustiveness of the docking global search, roughly proportional to computation time.

</br>

### Add Polar Hydrogens to Ligand / Receptor before Docking
- If enabled, this step will add Polar Hydrogens to the ligands / protein pockets before docking. Gasteiger charges will also be applied.

</br>

### Remove Non-Standard Amino Acid Residuals from Receptor before Docking
- If enabled, this step will remove any residue from any chain in the protein pockets before docking if its name is not in the below:

| CYS | ILE | SER | VAL |
|-----|-----|-----|-----|
| GLN | LYS | ASN | PRO |
| THR | PHE | ALA | HIS |
| GLY | ASP | LEU | ARG |
| TRP | GLU | TYR | MET |
| HID | HSP | HIE | HIP |
| CYX | CSS |     |     |

</br>

### Random Seed
- Sets the initial seed value for randomizers used in the Docking Analysis. 
- Different seeds can result in different variations of the docked ligand poses, even if the same model and protein pocket files are used. 
- Random seed is strictly a positive integer no larger than <code>4,294,967,295</code>.

</br>

### Cleanup Intermediate Files after Docking
- If enabled, cleans up the ligand and protein pockets' intermediate conversion files during preprocessing for Docking Analysis.

</br>

---

### Math Equations

To understand the model’s output, we use the following equations:

$$ E = mc^2 $$

This is Einstein’s famous equation for energy-mass equivalence.

More complex equations can also be represented like so:

$$ \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} $$

Where:
- $\mathbf{E}$ is the electric field
- $\rho$ is the charge density
- $\epsilon_0$ is the permittivity of free space.

This sentence uses delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$

Feel free to adjust the inputs and generate results based on your needs.