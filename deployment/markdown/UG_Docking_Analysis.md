<hr style="border: 0; border-top: 1px solid rgb(63,63,70);">

<p style="font-size: 16px; opacity: 0.9;">Ligand and Protein Pocket Preprocessing will be performed using <a href="https://ccsb.scripps.edu/mgltools/downloads/" style="color: rgba(249,115,21);"><b style="color: rgba(249,115,21);">MGLTools</b></a> while Docking Analysis will be performed using <a href="https://github.com/QVina/qvina/tree/master" style="color: rgba(249,115,21);"><b style="color: rgba(249,115,21);">QuickVina2.1</b></a>. Some default preprocessing steps are performed regardless of the selected options:</p>

```
Preprocessing Details

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
```

</br>

<h2><b>Ligand Fragment Size</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Graph Diffusion Models in general suffer from the issue of occasionally generating ligands with disconnected fragments.</p>
- <p style="font-size: 16px; opacity: 0.9;">This threshold filters ligands, retaining only those whose largest fully connected fragment size ≥ this specified size for docking. Ligands that don’t meet the threshold are excluded from docking.</p>
- <p style="font-size: 16px; opacity: 0.9;">A threshold of <code style="font-size: 14px;">0.0</code> includes all ligands' largest fragment, regardless of fragment size, while a threshold of <code style="font-size: 14px;">1.0</code> selects only fully connected ligands for docking.</p>

</br>

<h2><b>Search Space XYZ Dimensions (Angstroms)</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Dimensions of the cube-shaped docking search space in Angstroms.</p>

</br>

<h2><b>Search Exhaustiveness</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Controls the exhaustiveness of the docking global search. Higher values result in more thorough searches but require longer computation times.</p>

</br>

<h2><b>Add Polar Hydrogens to Ligand / Receptor before Docking</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Adds polar hydrogens to ligands or protein pockets before docking if enabled.</p>

</br>

<h2><b>Remove Non-Standard Amino Acid Residuals from Receptor before Docking</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">If enabled, this will remove any residue from any chain in the protein pockets before docking if it is not in the below 26 residues:</p>

</br>


<table style="width: 100%; border: 1px solid #ccc; border-collapse: collapse;">
  <thead style="background-color: rgba(249,115,21,0.5);">
    <tr>
      <th style="width: 10%; padding: 8px; text-align: center;">Residue</th>
      <th style="width: 40%; padding: 8px; text-align: center;">Full Name</th>
      <th style="width: 10%; padding: 8px; text-align: center;">Residue</th>
      <th style="width: 40%; padding: 8px; text-align: center;">Full Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>ALA</b></td>
      <td style="padding: 8px; text-align: left;  ">Alanine</td>
      <td style="padding: 8px; text-align: center;"><b>HIS</b></td>
      <td style="padding: 8px; text-align: left;  ">Histidine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>ARG</b></td>
      <td style="padding: 8px; text-align: left;  ">Arginine</td>
      <td style="padding: 8px; text-align: center;"><b>HSP</b></td>
      <td style="padding: 8px; text-align: left;  ">HIP, doubly protonated Histidine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>ASN</b></td>
      <td style="padding: 8px; text-align: left;  ">Asparagine</td>
      <td style="padding: 8px; text-align: center;"><b>ILE</b></td>
      <td style="padding: 8px; text-align: left;  ">Isoleucine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>ASP</b></td>
      <td style="padding: 8px; text-align: left;  ">Aspartic acid</td>
      <td style="padding: 8px; text-align: center;"><b>LEU</b></td>
      <td style="padding: 8px; text-align: left;  ">Leucine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>CSS</b></td>
      <td style="padding: 8px; text-align: left;  ">Cysteine disulfide (a covalently linked Cysteine residue in disulfide bonds)</td>
      <td style="padding: 8px; text-align: center;"><b>LYS</b></td>
      <td style="padding: 8px; text-align: left;  ">Lysine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>CYS</b></td>
      <td style="padding: 8px; text-align: left;  ">Cysteine</td>
      <td style="padding: 8px; text-align: center;"><b>MET</b></td>
      <td style="padding: 8px; text-align: left;  ">Methionine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>CYX</b></td>
      <td style="padding: 8px; text-align: left;  ">Cysteine in a disulfide bond (alternate representation)</td>
      <td style="padding: 8px; text-align: center;"><b>PHE</b></td>
      <td style="padding: 8px; text-align: left;  ">Phenylalanine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>GLN</b></td>
      <td style="padding: 8px; text-align: left;  ">Glutamine</td>
      <td style="padding: 8px; text-align: center;"><b>PRO</b></td>
      <td style="padding: 8px; text-align: left;  ">Proline</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>GLU</b></td>
      <td style="padding: 8px; text-align: left;  ">Glutamic acid</td>
      <td style="padding: 8px; text-align: center;"><b>SER</b></td>
      <td style="padding: 8px; text-align: left;  ">Serine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>GLY</b></td>
      <td style="padding: 8px; text-align: left;  ">Glycine</td>
      <td style="padding: 8px; text-align: center;"><b>THR</b></td>
      <td style="padding: 8px; text-align: left;  ">Threonine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>HID</b></td>
      <td style="padding: 8px; text-align: left;  ">Histidine (protonated at the delta nitrogen, Nδ)</td>
      <td style="padding: 8px; text-align: center;"><b>TRP</b></td>
      <td style="padding: 8px; text-align: left;  ">Tryptophan</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>HIE</b></td>
      <td style="padding: 8px; text-align: left;  ">Histidine (protonated at the epsilon nitrogen, Nε)</td>
      <td style="padding: 8px; text-align: center;"><b>TYR</b></td>
      <td style="padding: 8px; text-align: left;  ">Tyrosine</td>
    </tr>
    <tr>
      <td style="padding: 8px; text-align: center;"><b>HIP</b></td>
      <td style="padding: 8px; text-align: left;  ">Histidine (doubly protonated, both Nδ and Nε are protonated)</td>
      <td style="padding: 8px; text-align: center;"><b>VAL</b></td>
      <td style="padding: 8px; text-align: left;  ">Valine</td>
    </tr>
  </tbody>
</table>
</br>

<h2><b>Random Seed</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Sets the initial seed value for randomizers in Docking Analysis.</p>
- <p style="font-size: 16px; opacity: 0.9;">Different seeds yield variations in docked ligand poses, even with identical models and input files.</p>
- <p style="font-size: 16px; opacity: 0.9;"><b>Requirements</b>: Must be a positive integer ≤ <code style="font-size: 14px;">4,294,967,295</code>.</p>

</br>

<h2><b>Cleanup Intermediate Files after Docking</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Deletes intermediate conversion files for ligands and protein pockets after Docking Analysis if enabled.</p>

</br>
