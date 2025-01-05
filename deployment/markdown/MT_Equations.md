<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Atom Stability</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">01</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Measures the proportion of atoms that have the right valency. If two atoms have a distance shorter than the typical bond length plus the margin for the respective bond type, the atoms are considered to have a bond between them. After all bonds have been created, an atom is stable if its valency is precisely equal to the allowed number of bonds. <b>NOTE</b> that this metric does not take into account more atypical distances or aromatic bonds, but still measures whether the model is positioning the atoms precisely enough.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

$$ S_{Atom} = \frac{N_{Stable Bonds}}{N_{Atoms}} $$

Where:
- $N_{Stable Bonds}$ is the count of chemically stable bonds according to chemical rules.
- $N_{Atoms}$ is the count of all atoms in the list of stable molecules.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Molecule Stability</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">02</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Measures the proportion of generated molecules for which all atoms are stable (have the right valency). However, this metric yields near <code style="font-size: 14px;">0.0</code> for both the <b>GEOM-Drugs</b> and <b>CrossDocked2020</b> dataset as they both contain molecules of large sizes, which introduces more atypical behavior.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

$$ S_{Mol} = \frac{N_{s}}{N_{g}} $$

Where:
- $N_{s}$ is the count of generated molecules where all bonds are stable according to chemical rules.
- $N_{g}$ is the total number of generated molecules.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Validity</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">03</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Measures the proportion of chemically valid molecules (can successfully be converted to RDKit Molecules) among all the generated molecules. All computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

$$ V = \frac{N_{v}}{N_{g}} $$

Where:
- $N_{v}$ is the number of valid molecules that can be successfulyl converted into RDKit Molecules.
- $N_{g}$ is the total number of generated molecules.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Uniqueness</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">04</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Uniqueness is defined as the ratio between the number of unique molecules and valid molecules, it measures the degree of variety in the valid molecules generated. Molecules are first converted from graph representation into canonical SMILES string representation for this computation. All computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

$$ U = \frac{N_{u}}{N_{v}} $$

Where:
- $N_{u}$ is the count of unique valid molecules (each having a distinct SMILES representation).
- $N_{v}$ is the number of valid molecules that can be successfully converted to SMILES notation.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Diversity</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">05</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Diversity measures the richness in variation of the generated molecules. The <code>RDKit_Fingerprint</code> is first computed for all molecules. All computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

$$ D = 1 - SIM_{Tanimoto}(F_{Mol 1}, F_{Mol 2}) $$

Where:
- $SIM_{Tanimoto}$ is the function to compute Tanimoto Similarity between 2 molecule fingerprints.
- $F_{Mol 1}$ is the RDKit Fingerprint for Molecule A.
- $F_{Mol 2}$ is the RDKit Fingerprint for Molecule B.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Connectivity</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">06</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Measures if a molecule is in one single connected piece. It is calculated by taking a division between the number of atoms in the largest fragment of the molecule by the total number of atoms. Extraction of atoms and fragments are done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

- <p style="font-size: 16px; opacity: 0.9;"><code>0.0</code> suggests that all molecules consist of multiple disconnected fragments.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>1.0</code> indicates all molecules are fully connected.</p>

$$ C = \frac{N_{Atoms In Largest Fragment}}{N_{All Atoms}} $$

Where:
- $N_{Atoms In Largest Fragment}$ is the number of atoms in the largest fragment of the molecule.
- $N_{All Atoms}$ is the total number of atoms in the molecule.

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>QED (Quantitative Estimate of Drug-Likeliness)</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">07</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Represents how "drug-like" a molecule is, based on a set of desirable properties commonly found in approved drugs. Factors considered when determining the QED of a molecule are Molecular Weight, LogP, Topological Polar Surface Area, Number of Hydrogen Bond Donors and Acceptors, Number of Aromatic Rings, Number of Rotatable Bonds, and etc.</p>

<p style="font-size: 16px; opacity: 0.9;">Computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

- <p style="font-size: 16px; opacity: 0.9;"><code>0.0</code> means that the molecule is highly drug-like.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>1.0</code> means that the molecule is unlikely to be a viable drug candidate.</p>

</br>

```
💡 NOTE: QED should be regarded as a reference metric, as it is based on the statistical properties of traditional drugs designed specifically for oral administration.
```

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>SA (Synthetic Accessibility Score)</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">08</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Evaluates how easy or difficult it would be to synthesize a molecule in a laboratory setting. Factors considered when determining the QED of a molecule are Molecular Complexity, Presence of Unusual Chemical Groups, Ring Systems, Number of Rotatable Bonds, and etc.</p>

<p style="font-size: 16px; opacity: 0.9;">Computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library and code released by the Novartis Institutes for BioMedical Research Inc.</p>

<code>Value Range: [0, 1]. The higher the better.</code>

- <p style="font-size: 16px; opacity: 0.9;"><code>0.0</code> represents molecules that are very difficult to synthesize.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>1.0</code> represents molecules that are very easy to synthesize.</p>

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>LogP</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">09</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Estimates the hydrophobicity of a molecule - essential for predicting drug absorption, distribution, and bioavailability. Computed as the logarithm (base 10) of the <b>ratio</b> of a compound's <b>concentration in a nonpolar (lipid) phase</b> (typically octanol) to its <b>concentration in a polar (aqueous) phase</b> (typically water). Computation done using the python <a href="https://www.rdkit.org/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">RDKit</b></a> library.</p>

<p style="font-size: 16px; opacity: 0.9;">Descriptions for different LogP values are as follows:</p>

- <p style="font-size: 16px; opacity: 0.9;"><code>LogP < 0</code>: molecule is hydrophilic, may have poor membrane permeability.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>LogP ≈ 0</code>: molecule has a balanced distribution between aqueous and lipid phases.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>1 < LogP < 3</code>: Often considered optimal for many drugs, balancing membrane permeability and solubility.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>LogP > 3</code>: Enhanced membrane permeability but may also lead to poor solubility in water, increased risk of nonspecific binding, and potential toxicity due to accumulation in lipid-rich tissues.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>LogP > 5</code>: Might cause increased metabolic instability, and potential toxicity.</p>

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Lipinski (Lipinski’s Rule of Five)</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">10</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">Measures how many rules in the Lipinski rule of five are satisfied (a loose rule of thumb to assess the drug-likeness of molecules).</p>

<p style="font-size: 16px; opacity: 0.9;">The Lipinski’s Rule of Five are:</p>

1. <p style="font-size: 16px; opacity: 0.9;"><code>Molecular Weight < 500 Da</code>: Molecules larger than 500 Da often have difficulty being absorbed in the gastrointestinal tract, reducing their bioavailability.</p>
2. <p style="font-size: 16px; opacity: 0.9;"><code>LogP ≤ 5</code>: A logP value of 5 or less indicates that the compound has balanced hydrophilic and lipophilic properties, which helps with absorption and transport across membranes.</p>
3. <p style="font-size: 16px; opacity: 0.9;"><code>No. Hydrogen Bond Donors ≤ 5</code>: Having more than 5 hydrogen bond donors (i.e. OH or NH groups) increases a molecule’s solubility but can make it less permeable to cell membranes.</p>
4. <p style="font-size: 16px; opacity: 0.9;"><code>No. Hydrogen Bond Acceptors ≤ 10</code>: Molecules with more than 10 hydrogen bond acceptors (e.g., N and O atoms) tend to have reduced permeability across cell membranes.</p>
5. <p style="font-size: 16px; opacity: 0.9;"><code>No. Rotatable Bonds ≤ 10</code>: Too many rotatable bonds can lead to increased flexibility in a molecule, making it more difficult to bind to the target receptor.</p>

</br>

<code>Value Range: [0, 5]. The higher the better.</code>

- <p style="font-size: 16px; opacity: 0.9;"><code>0</code> means that none of the above rules are satisfied.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>5</code> means that the molecule is highly likely to be drug-like and suitable for oral administration based on Lipinski's rules.</p>

</br>

```
💡 NOTE: Molecules that satisfy < 5 rules aren’t necessarily bad candidates, some approved drugs violate one or more of these rules but are still effective. This metric should also be regarded as a reference metric, as its statistics were derived from traditional drugs designed specifically for oral administration.
```

</br>
</br>

<hr style="border: 0; border-top: 1px solid rgb(63,63,70); margin: 25px 0">

<div style="display: flex; justify-content: space-between; align-items: baseline;">
    <h2><b>Qvina (Binding Affinity)</b></h2>
    <h1 style="opacity: 0.5; font-size: 60px;">11</h1>
</div>

<p style="font-size: 16px; opacity: 0.9;">A measure of how strongly the ligand binds to the receptor at the docked position. It reflects the <b>amount of energy required to bind or release a single mole of ligand molecules to/from a receptor</b>. Lower energy values (more negative) indicate stronger binding and higher stability of the ligand-receptor complex. It is calculated as the change in free energy of binding $\Delta G$, which includes contributions from various interactions:</p>

- <p style="font-size: 16px; opacity: 0.9;">Hydrogen bonds</p>
- <p style="font-size: 16px; opacity: 0.9;">Van der Waals forces</p>
- <p style="font-size: 16px; opacity: 0.9;">Electrostatic interactions</p>
- <p style="font-size: 16px; opacity: 0.9;">Desolvation effects</p>
- <p style="font-size: 16px; opacity: 0.9;">Entropy changes (approximated)</p>

</br>

<p style="font-size: 16px; opacity: 0.9;">Ligand and Protein Pocket Preprocessing are done using <a href="https://ccsb.scripps.edu/mgltools/downloads/" style="color: rgb(249,115,21);"><b style="color: rgb(249,115,21);">MGLTools</b></a> while Docking Analysis is done using <a href="https://github.com/QVina/qvina/tree/master" style="color: rgba(249,115,21);"><b style="color: rgba(249,115,21);">QuickVina2.1</b></a>.</p>

<code>Typical Ranges:</code>
- <p style="font-size: 16px; opacity: 0.9;"><code>-9 to -14 kcal/mol</code>: High binding affinity (i.e. strong inhibitors or drugs).</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>-5 to -8 kcal/mol</code>: Moderate binding affinity.</p>
- <p style="font-size: 16px; opacity: 0.9;"><code>> -4 kcal/mol</code>: Weak or negligible binding.</p>

</br>
</br>