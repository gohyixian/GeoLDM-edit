<hr style="border: 0; border-top: 1px solid rgb(63,63,70);">

<h2><b>Number of Ligand Samples to Generate per Pocket</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">Specifies how many ligand samples to generate for each uploaded protein pocket.</p>

</br>

<h2><b>Randomly Sample Number of Atoms per Ligand from Pre-computed Distribution</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">When enabled (<code style="font-size: 14px;">True</code>), the number of atoms (excluding hydrogen) in each generated ligand is randomly sampled from a pre-computed distribution derived from the model's training dataset.</p>

- <p style="font-size: 16px; opacity: 0.9;">This distribution reflects the typical atom counts observed during training and is visualized as a plot for better insight.</p>

</br>

<h2><b>Delta Number of Atoms per Ligand</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">This parameter applies only when random sampling of atom counts is enabled.</p>
- <p style="font-size: 16px; opacity: 0.9;">Allows you to fine-tune the atom count for each generated ligand by applying an increase to the sampled value. The final number of atoms is given by:</p>
</br>

$$N_{final} = N_{sampled} + N_{\Delta}$$

<h2><b>Number of Atoms per Ligand</b></h2>

- <p style="font-size: 16px; opacity: 0.9;">This option is only available when the random sampling of number of atoms is disabled.</p>
- <p style="font-size: 16px; opacity: 0.9;">Allows you to manually specify the number of atoms (excluding hydrogen) for each generated ligand.</p>

</br>
