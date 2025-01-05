<h1><b style='color:#f97315; font-size: 36px;'>User Guidelines</b></h1>


<p style="font-size: 16px;">Welcome to the Control-GeoLDM model interface. Below are some guidelines to help you get started:</p>

<h2><b style='color:white; font-size: 24px;'>1. Upload Protein Pocket Files</b></h2>

- <p style="font-size: 16px;">Upload one or more protein pocket files in PDB format.</p>
- <p style="font-size: 16px;">Drag and drop files or use the upload button to select them.</p>

</br>

<h2><b style='color:white;'>2. Download Results</b></h2>

- <p style="font-size: 16px;">Once generation is complete, you may download the generated ligand SDF files via the provided zip file. If Docking is performed, the zipped file will include docked ligands and protein complexes in PDBQT files.</p>

</br>

<h2><b style='color:white;'>3. View Metrics</b></h2>

- <p style="font-size: 16px;">You may access detailed metrics for the generated ligands in the <code style="font-size: 14px;">Metrics</code> tab. Metrics are also included in the downloadable results file for offline review.</p>


</br>

<h1 style='color:#f97315; font-size: 36px;'>Model Configurations</h1>

<h2><b style='color:white;'>Model Selection</b></h2>

- You may choose a model from the dropdown menu.
- **Models with <code>CA</code> in their names**: These models extract only the Alpha Carbon (CA) atoms of each residue in the protein pocket, using them to represent the pocket during ligand generation.
- **Models without <code>CA</code> in their names**: These models use the full atomic representation of the protein pocket for generation.

</br>

<h2><b style='color:white;'>Random Seed</b></h2>

- The random seed determines the starting value for all randomizers used by the program, excluding Docking Analysis.
- Using different random seed values can produce variations in the generated ligands, even with the same model and input files.
- **Requirements**: The seed must be a positive integer ≤ <code>4,294,967,295</code>.

</br>

<h2><b style='color:white;'>Batch Size</b></h2>

- Batch size specifies the number of ligands generated in a single run.
- A larger batch size speeds up computation while having zero effect on the generated ligands. 
- The maximum batch size depends on the GPU's capacity and varies by model.



</br>

<h1 style='color:#f97315; font-size: 36px;'>Ligand Generation Configurations</h1>

<h2><b style='color:white;'>Number of Ligand Samples to Generate per Pocket</b></h2>

- Specifies how many ligand samples to generate for each uploaded protein pocket.

</br>

<h2><b style='color:white;'>Randomly Sample Number of Atoms per Ligand from Pre-computed Distribution</b></h2>

- **Description**: When enabled (<code>True</code>), the number of atoms (excluding hydrogen) in each generated ligand is randomly sampled from a pre-computed distribution derived from the model's training dataset.
- **Purpose**: This distribution reflects the typical atom counts observed during training and is visualized as a plot for better insight.


</br>

<h2><b style='color:white;'>Delta Number of Atoms per Ligand</b></h2>

- This parameter applies only when random sampling of atom counts is enabled.
- Allows you to fine-tune the atom count for each generated ligand by applying an increase to the sampled value. The final number of atoms is given by:

$$N_{final} = N_{sampled} + N_{\Delta}$$
​

<h2><b style='color:white;'>Number of Atoms per Ligand</b></h2>

- This option is only available when the random sampling of number of atoms is disabled.
- Allows you to manually specify the number of atoms (excluding hydrogen) for each generated ligand.



</br>

<h1 style='color:#f97315; font-size: 36px;'>Docking Analysis Configurations</h1>

Docking Analysis will be performed using [QuickVina2.1](https://github.com/QVina/qvina/raw/master/bin/). Some default preprocessing steps are performed regardless of the selected options:

<h2><b style='color:white;'>Preprocessing Details</b></h2>

Ligand: 
- Apply Gasteiger charges.
- Merge charges and remove non-polar hydrogens.
- Merge charges and remove lone pairs.

Protein Pockets:
- Apply Gasteiger charges.
- Merge charges and remove non-polar hydrogens.
- Merge charges and remove lone pairs.
- Remove water residues.
- Remove chains composed entirely of residues types other than the standard 20 amino acids.

</br>

<h2><b style='color:white;'>Molecule Fragment Size</b></h2>

- Filters and retains generated ligands for docking analysis based on their largest fully connected fragment size.
- Set the threshold to 0 to automatically select the largest fully connected fragment in each molecule (regardless of size) for docking analysis.

</br>

<h2><b style='color:white;'>Search Space XYZ Dimensions (Angstroms)</b></h2>

- Dimensions of the cube-shaped docking search space in Angstroms.

</br>

<h2><b style='color:white;'>Search Exhaustiveness</b></h2>

- Controls the exhaustiveness of the docking global search. Higher values result in more thorough searches but require longer computation times.

</br>

<h2><b style='color:white;'>Add Polar Hydrogens to Ligand / Receptor before Docking</b></h2>

- Adds polar hydrogens to ligands or protein pockets before docking if enabled.

</br>

<h2><b style='color:white;'>Remove Non-Standard Amino Acid Residuals from Receptor before Docking</b></h2>

- If enabled, this will remove any residue from any chain in the protein pockets before docking if it is not in the below 26 residues:

</br>

| Residue | Full Name                                                                    | Residue | Full Name                        |
|---------|------------------------------------------------------------------------------|---------|----------------------------------|
| ALA     | Alanine                                                                      | HIS     | Histidine                        |
| ARG     | Arginine                                                                     | HSP     | HIP, doubly protonated Histidine |
| ASN     | Asparagine                                                                   | ILE     | Isoleucine                       |
| ASP     | Aspartic acid                                                                | LEU     | Leucine                          |
| CSS     | Cysteine disulfide (a covalently linked Cysteine residue in disulfide bonds) | LYS     | Lysine                           |
| CYS     | Cysteine                                                                     | MET     | Methionine                       |
| CYX     | Cysteine in a disulfide bond (alternate representation)                      | PHE     | Phenylalanine                    |
| GLN     | Glutamine                                                                    | PRO     | Proline                          |
| GLU     | Glutamic acid                                                                | SER     | Serine                           |
| GLY     | Glycine                                                                      | THR     | Threonine                        |
| HID     | Histidine (protonated at the delta nitrogen, Nδ)                             | TRP     | Tryptophan                       |
| HIE     | Histidine (protonated at the epsilon nitrogen, Nε)                           | TYR     | Tyrosine                         |
| HIP     | Histidine (doubly protonated, both Nδ and Nε are protonated)                 | VAL     | Valine                           |



</br>

<h2><b style='color:white;'>Random Seed</b></h2>

- Sets the initial seed value for randomizers in Docking Analysis.
- Different seeds yield variations in docked ligand poses, even with identical models and input files.
- **Requirements**: Must be a positive integer ≤ 4,294,967,295.
</br>

<h2><b style='color:white;'>Cleanup Intermediate Files after Docking</b></h2>

- Deletes intermediate conversion files for ligands and protein pockets after Docking Analysis if enabled.

</br>
