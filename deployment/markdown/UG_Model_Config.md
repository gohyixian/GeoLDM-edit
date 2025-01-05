<hr style="border: 0; border-top: 1px solid rgb(63,63,70);">

<h2><b>Model Selection</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">You may choose a model from the dropdown menu.</p>
- <p style="font-size: 16px; opacity: 0.9;"><b>Models with <code>CA</code> in their names</b>: These models extract only the Alpha Carbon (CA) atoms of each residue in the protein pocket, using them to represent the pocket during ligand generation.</p>
- <p style="font-size: 16px; opacity: 0.9;"><b>Models without <code>CA</code> in their names</b>: These models use the full atomic representation of the protein pocket for generation.</p>

</br>

<h2><b>Random Seed</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">The random seed determines the starting value for all randomizers used by the program, excluding Docking Analysis.</p>
- <p style="font-size: 16px; opacity: 0.9;">Using different random seed values can produce variations in the generated ligands, even with the same model and input files.</p>
- <p style="font-size: 16px; opacity: 0.9;"><b>Requirements</b>: The seed must be a positive integer ≤ <code style="font-size: 14px;">4,294,967,295</code>.</p>

</br>

<h2><b>Batch Size</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Batch size specifies the number of ligands generated in a single run.</p>
- <p style="font-size: 16px; opacity: 0.9;">A larger batch size speeds up computation while having zero effect on the generated ligands. </p>
- <p style="font-size: 16px; opacity: 0.9;">The maximum batch size depends on the GPU's capacity and varies by model.</p>

</br>
