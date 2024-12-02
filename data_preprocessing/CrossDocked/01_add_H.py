"""
This script uses UCSF ChimeraX to add hydrogens to each ligand molecule.

To run this script, open up ChimeraX, and run the below:

    runscript /path/to/this/file/01_add_H.py
"""

import os
from chimerax.core.commands import run

def process_sdf_files(session, input_dir, output_dir):
    """
    Process .sdf files in a directory: add hydrogens and save as .pdb files.

    Args:
        session: ChimeraX session object.
        input_dir (str): Path to the input directory containing .sdf files.
        output_dir (str): Path to the output directory for .pdb files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    counter = 0
    
    # Iterate over all .sdf files in the input directory
    for file_name in sorted(list(os.listdir(input_dir)))[50]:
        if file_name.endswith(".sdf"):
            input_path = os.path.join(input_dir, file_name)
            base_name = os.path.splitext(file_name)[0]  # File name without extension
            output_path = os.path.join(output_dir, f"{base_name}.pdb")

            # Open the .sdf file
            run(session, f"open {input_path}")
            
            # Add hydrogen atoms
            run(session, "addh")

            # Save the structure as a .pdb file
            run(session, f"save {output_path} format pdb")

            # Close the structure to free memory
            run(session, "close all")
            
            print(f"Processed file {counter}: {file_name}")
            session.logger.info(f"Processed file {counter}: {file_name}")

            counter += 1

    print(f"Processed all files. Output saved to {output_dir}")

# Example usage:
# Replace 'your_input_directory' and 'your_output_directory' with actual paths
# process_sdf_files(session, "your_input_directory", "your_output_directory")

process_sdf_files(session, "C:\Users\PC\Desktop\yixian\Crossdocked_Pocket10_ToAddH\data", "C:\Users\PC\Desktop\yixian\Crossdocked_Pocket10_ToAddH\data_add_H")

# runscript C:\Users\PC\Desktop\yixian\GeoLDM-edit\data_preprocessing\CrossDocked\01_add_H.py