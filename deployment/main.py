import os
import time
import shutil
import argparse
import gradio as gr
from pathlib import Path
import plotly.graph_objects as go

from deployment.modules.controlnet import init_model_and_sample
from deployment.modules.errors import (
    InvalidInputError,
    ModelInitialisationError,
    ModelGenerationError,
    MetricError
)
from deployment.utils import (
    zip_folder, 
    get_empty_metrics_df,
    get_available_models, 
    approximate_max_batch_size, 
    get_model_n_nodes_distribution
)


parser = argparse.ArgumentParser(description='Control-GeoLDM Deployment')
parser.add_argument('-u', '--public', action='store_true',help='Spits out a public URL that can be accessed for 72 hours.')
parser.add_argument('-a', '--server_address', type=str, default='127.0.0.1' ,help='Server address')
parser.add_argument('-p', '--server_port', type=int, default=7860 ,help='Server port')
opt = parser.parse_args()

# model directory
MODEL_ZOO = "./deployment/models/controlnet"

# temprorary processing dir
TEMP_DIR = "./deployment/tmp"

# mgltools env name
MGLTOOLS_ENV_NAME = "mgltools-python2"

# UI elements
ELEMENT_MIN_WIDTH_PX = 150
TAB_TITLE   = "Control-GeoLDM"
TAB_FAVICON = "./deployment/assets/images/pacman.png"

# maximums
MAX_NUM_LIGANDS_PER_POCKET = 150
MAX_DELTA_NUM_ATOMS        = 20
MAX_NUM_ATOMS_PER_LIGAND   = 100
MAX_QVINA_SEARCH_SIZE      = 40
MAX_QVINA_EXHAUSTIVITY     = 40

# allowed pocket file formats
ALLOWED_POCKET_FORMATS = [".pdb"]

# user guideline markdown
UG_GENERAL_MD           = "./deployment/assets/markdown/UG_General.md"
UG_MODEL_CONFIG_MD      = "./deployment/assets/markdown/UG_Model_Config.md"
UG_LIGAND_GENERATION_MD = "./deployment/assets/markdown/UG_Ligand_Generation.md"
UG_DOCKING_ANALYSIS_MD  = "./deployment/assets/markdown/UG_Docking_Analysis.md"
MT_QEUATIONS_MD         = "./deployment/assets/markdown/MT_Equations.md"

# override gradio's default LaTeX delimiters for math equations
LATEX_DELIMITERS = \
    [
        {"left": "$$", "right": "$$", "display": True},  # center
        {"left": "$", "right": "$", "display": False}    # inline
    ]


# get available models, approximate maximum batch size 
# for each model on current device, and get each model's 
# training data's precomputed distribution of number of atoms per ligand
AVAILABLE_MODELS = get_available_models(MODEL_ZOO)
AVAILABLE_MODELS = approximate_max_batch_size(AVAILABLE_MODELS)
AVAILABLE_MODELS = get_model_n_nodes_distribution(AVAILABLE_MODELS)

# metrics df
METRICS_DF = get_empty_metrics_df()


# pre-read markdown content
with open(UG_GENERAL_MD, "r") as f:
    ug_general_content = f.read()
with open(UG_MODEL_CONFIG_MD, "r") as f:
    ug_model_config_content = f.read()
with open(UG_LIGAND_GENERATION_MD, "r") as f:
    ug_ligand_generation_content = f.read()
with open(UG_DOCKING_ANALYSIS_MD, "r") as f:
    ug_docking_analysis_content = f.read()
with open(MT_QEUATIONS_MD, "r") as f:
    mt_equations_content = f.read()


def plot_histogram(
    data_dict: dict[int, int], 
    delta_atoms: int=0
) -> go.Figure:
    
    x_values = list(data_dict.keys())
    y_values = list(data_dict.values())

    x_values = [x + delta_atoms for x in x_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='#f97216', width=3),
        fill='tozeroy',
        name="Density"
    ))
    
    fig.update_layout(
        # title="Pre-computed Number of Atoms per Ligand Distribution",
        xaxis_title="Number of Atoms per Ligand",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(128,128,128)'),
        xaxis=dict(
            linecolor='rgb(128,128,128)',
            gridcolor='rgba(128,128,128,0.25)',
            range=[0, max(list(data_dict.keys())) + MAX_DELTA_NUM_ATOMS]
        ),
        yaxis=dict(
            linecolor='rgb(128,128,128)',
            gridcolor='rgba(128,128,128,0.25)',
        ),
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def toggle_random_num_gen_visibility(is_checked):
    return [
        gr.update(visible=is_checked),      # plot
        gr.update(visible=is_checked),      # delta atoms
        gr.update(visible=not is_checked)   # specific num atoms
    ]

def toggle_docking_analysis_visibility(is_checked):
    return [
        gr.update(visible=is_checked),   # connectivity_threshold
        gr.update(visible=is_checked),   # ligand_add_h
        gr.update(visible=is_checked),   # receptor_add_h
        gr.update(visible=is_checked),   # remove_nonstd_resi
        gr.update(visible=is_checked),   # qvina_size
        gr.update(visible=is_checked),   # qvina_exhaustiveness
        gr.update(visible=is_checked),   # qvina_seed
        gr.update(visible=is_checked)    # cleanup_files
    ]

def update_bs_n_nodes_plot(selected_model, delta_atoms_value, batch_size):
    return [
        plot_histogram(   # n_nodes_plot
            AVAILABLE_MODELS[selected_model]['n_nodes'], 
            delta_atoms_value
        ),
        gr.update(        # batch_size
            maximum=AVAILABLE_MODELS[selected_model]['max_bs'], 
            value=min(batch_size, AVAILABLE_MODELS[selected_model]['max_bs'])
        ),
    ]



def main_script(
    pdb_files,
    model_selected: str,
    model_seed: int = 42,
    model_batch_size: int = 60,
    num_ligands_per_pocket: int = 1,
    sample_num_atoms_per_ligand: bool = True,
    delta_num_atoms_per_ligand: int = 5,
    specific_num_atoms_per_ligand: int = 30,
    compute_qvina: bool = True,
    qvina_connectivity_thres: float = 1.,
    qvina_size: int = 20,
    qvina_exhaustiveness: int = 16,
    qvina_ligand_add_H: bool = False,
    qvina_receptor_add_H: bool = False,
    qvina_remove_nonstd_resi: bool = False,
    qvina_seed: int = 42,
    qvina_cleanup_files: bool = True
):
    # remove dir from previous run
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    # name for this run
    run_name = f"{time.time_ns()}_{model_selected}"
    
    # create tmp dir for pocket files
    pocket_temp_dir = str(Path(TEMP_DIR, "pocket_pdb_files"))
    if not os.path.exists(pocket_temp_dir):
        os.makedirs(pocket_temp_dir)

    # create tmp dir for results
    results_temp_dir = str(Path(TEMP_DIR, run_name))
    if not os.path.exists(results_temp_dir):
        os.makedirs(results_temp_dir)
    
    try:
        # copy pocket files
        for file in pdb_files:
            shutil.copy(file, str(Path(pocket_temp_dir, Path(file).name)))
    except:
        return None, get_empty_metrics_df(), "No Input Files Provided!"
    
    try:
        # load model, sample, and compute metrics
        METRICS_DF = init_model_and_sample(
            pocket_pdb_dir=pocket_temp_dir,
            results_path=results_temp_dir,
            model_dict=AVAILABLE_MODELS[model_selected],
            model_seed=model_seed,
            model_batch_size=model_batch_size,
            num_ligands_per_pocket=num_ligands_per_pocket,
            sample_num_atoms_per_ligand=sample_num_atoms_per_ligand,
            delta_num_atoms_per_ligand=delta_num_atoms_per_ligand,
            specific_num_atoms_per_ligand=specific_num_atoms_per_ligand,
            compute_qvina=compute_qvina,
            qvina_connectivity_thres=qvina_connectivity_thres,
            qvina_size=qvina_size,
            qvina_exhaustiveness=qvina_exhaustiveness,
            qvina_ligand_add_H=qvina_ligand_add_H,
            qvina_receptor_add_H=qvina_receptor_add_H,
            qvina_remove_nonstd_resi=qvina_remove_nonstd_resi,
            qvina_seed=qvina_seed,
            qvina_cleanup_files=qvina_cleanup_files,
            mgltools_env_name=MGLTOOLS_ENV_NAME
        )
    except InvalidInputError as e:
        return None, get_empty_metrics_df(), "Invalid Input Files!"
    except ModelInitialisationError as e:
        return None, get_empty_metrics_df(), "Model Initialisation Error!"
    except ModelGenerationError as e:
        return None, get_empty_metrics_df(), "Model Generation Error!"
    except MetricError as e:
        return None, get_empty_metrics_df(), "Metric Computation / Docking Analysis Error!"
    except Exception as e:
        return None, get_empty_metrics_df(), "General Error!"
    
    # zip_results
    zip_filename = str(Path(TEMP_DIR, f"{run_name}.zip"))
    zip_folder(results_temp_dir, zip_filename)
    
    return zip_filename, METRICS_DF, ""


with gr.Blocks(title=TAB_TITLE) as app:
    gr.Markdown("# Control-GeoLDM")
    with gr.Tab("Generation"):
        
        # Error message component
        error_popup = gr.Textbox(visible=False, label="Error", interactive=False, container=True)
        
        with gr.Row():
            with gr.Column(scale=4, min_width=4*ELEMENT_MIN_WIDTH_PX):
                with gr.Row():
                    pdb_files = gr.File(label="Upload Protein Pocket PDB Files", file_count="multiple", file_types=ALLOWED_POCKET_FORMATS)
                    output_zip_file = gr.File(label="Download Result Files", interactive=True)
            
                with gr.Row():
                    generate_button = gr.Button("Generate")
                    

        # </br>
        with gr.Row():
            pass
        with gr.Row():
            pass


        gr.Markdown("## Model Configurations")
        with gr.Row():
            with gr.Column(scale=2, min_width=2*ELEMENT_MIN_WIDTH_PX):
                selected_model = gr.Dropdown(
                    choices=sorted(list(AVAILABLE_MODELS.keys())), 
                    value=sorted(list(AVAILABLE_MODELS.keys()))[0], 
                    label="Select Model"
                )
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                model_seed = gr.Number(label="Random Seed (Integer)", minimum=0, maximum=2**32 - 1, value=42)
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                batch_size = gr.Slider(
                    minimum=1, 
                    maximum=AVAILABLE_MODELS[sorted(list(AVAILABLE_MODELS.keys()))[0]]['max_bs'], 
                    step=1, 
                    value=AVAILABLE_MODELS[sorted(list(AVAILABLE_MODELS.keys()))[0]]['max_bs'], 
                    label="Batch Size"
                )


        # </br>
        with gr.Row():
            pass
        with gr.Row():
            pass


        gr.Markdown("## Ligand Generation Configurations")
        with gr.Row():
            num_samples_per_pocket = gr.Slider(minimum=1, maximum=MAX_NUM_LIGANDS_PER_POCKET, step=1, value=1, label="Number of Ligand Samples to Generate per Pocket")
        with gr.Row():
            random_atoms = gr.Checkbox(value=True, label="Randomly Sample Number of Atoms per Ligand from Pre-computed Distribution")
        with gr.Row():
            n_nodes_plot = gr.Plot(visible=True)
        with gr.Row():
            delta_atoms = gr.Slider(minimum=0, maximum=MAX_DELTA_NUM_ATOMS, step=1, value=5, label="Delta Number of Atoms per Ligand", visible=True)
        with gr.Row():
            specific_atoms = gr.Slider(minimum=1, maximum=MAX_NUM_ATOMS_PER_LIGAND, value=30, step=1, label="Number of Atoms per Ligand", visible=False)


        # </br>
        with gr.Row():
            pass
        with gr.Row():
            pass


        gr.Markdown("## Docking Analysis Configurations")
        with gr.Row():
            do_docking_analysis = gr.Checkbox(value=True, label="Perform Docking Analysis")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                connectivity_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.0, label="Ligand Fragment Size", visible=True)
            with gr.Column(scale=2, min_width=2*ELEMENT_MIN_WIDTH_PX):
                with gr.Row():
                    qvina_size = gr.Slider(minimum=1, maximum=MAX_QVINA_SEARCH_SIZE, value=20, label="Search Space XYZ Dimensions (Angstroms)", visible=True)
                    qvina_exhaustiveness = gr.Slider(minimum=1, maximum=MAX_QVINA_EXHAUSTIVITY, value=16, label="Search Exhaustiveness", visible=True)
            
        with gr.Row():
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                ligand_add_h = gr.Checkbox(value=True, label="Add Polar Hydrogens to Ligand before Docking", visible=True)
                receptor_add_h = gr.Checkbox(value=True, label="Add Polar Hydrogens to Receptor before Docking", visible=True)
                remove_nonstd_resi = gr.Checkbox(value=False, label="Remove Non-Standard Amino Acid Residuals from Receptor before Docking", visible=True)
        
        with gr.Row():
            qvina_seed = gr.Number(label="Random Seed (Integer)", minimum=0, maximum=2**32 - 1, value=42)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                cleanup_files = gr.Checkbox(value=True, label="Cleanup Intermediate Files after Docking", visible=True)
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                pass
            with gr.Column(scale=1, min_width=ELEMENT_MIN_WIDTH_PX):
                pass


    with gr.Tab("Metrics"):
        with gr.Row():
            results_table = gr.DataFrame(value=METRICS_DF, label="", max_height=1000)
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 style="color:#f97315; font-size: 36px;">Metrics' Definitions</h1>
            </div>
            """,
            latex_delimiters=LATEX_DELIMITERS
        )
        gr.Markdown(mt_equations_content, latex_delimiters=LATEX_DELIMITERS)


    with gr.Tab("User Guidelines"):
        gr.Markdown("<h1><b style='color:#f97315; font-size: 36px;'>User Guidelines</b></h1>", latex_delimiters=LATEX_DELIMITERS)
        gr.Markdown(ug_general_content, latex_delimiters=LATEX_DELIMITERS)
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 style="color:#f97315; font-size: 36px;">Model Configurations</h1>
                <h1 style="color:rgba(249,115,21,0.5); font-size: 75px;">01</h1>
            </div>
            """,
            latex_delimiters=LATEX_DELIMITERS
        )
        gr.Markdown(ug_model_config_content, latex_delimiters=LATEX_DELIMITERS)
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 style="color:#f97315; font-size: 36px;">Ligand Generation Configurations</h1>
                <h1 style="color:rgba(249,115,21,0.5); font-size: 75px;">02</h1>
            </div>
            """,
            latex_delimiters=LATEX_DELIMITERS
        )
        gr.Markdown(ug_ligand_generation_content, latex_delimiters=LATEX_DELIMITERS)
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h1 style="color:#f97315; font-size: 36px;">Docking Analysis Configurations</h1>
                <h1 style="color:rgba(249,115,21,0.5); font-size: 75px;">03</h1>
            </div>
            """,
            latex_delimiters=LATEX_DELIMITERS
        )
        gr.Markdown(ug_docking_analysis_content, latex_delimiters=LATEX_DELIMITERS)


    # run on app load
    app.load(fn=update_bs_n_nodes_plot, inputs=[selected_model, delta_atoms, batch_size], outputs=[n_nodes_plot, batch_size])
    
    # run on toggle change
    delta_atoms.change(
        fn=update_bs_n_nodes_plot, 
        inputs=[selected_model, delta_atoms, batch_size], 
        outputs=[n_nodes_plot, batch_size]
    )
    selected_model.change(
        fn=update_bs_n_nodes_plot, 
        inputs=[selected_model, delta_atoms, batch_size], 
        outputs=[n_nodes_plot, batch_size]
    )
    random_atoms.change(
        fn=toggle_random_num_gen_visibility,
        inputs=random_atoms,
        outputs=[n_nodes_plot, delta_atoms, specific_atoms]
    )
    do_docking_analysis.change(
        fn=toggle_docking_analysis_visibility,
        inputs=do_docking_analysis,
        outputs=[connectivity_threshold, ligand_add_h, receptor_add_h, remove_nonstd_resi, qvina_size, qvina_exhaustiveness, qvina_seed, cleanup_files]
    )


    def on_generate_click_handler(*args):
        zip_file, table, error = main_script(*args)
        error_visibility = bool(error)
        return zip_file, table, gr.update(visible=error_visibility, value=error)


    # run main script
    generate_button.click(
        on_generate_click_handler,
        inputs=[
            pdb_files,
            selected_model,
            model_seed,
            batch_size,
            num_samples_per_pocket,
            random_atoms,
            delta_atoms,
            specific_atoms,
            do_docking_analysis,
            connectivity_threshold,
            qvina_size,
            qvina_exhaustiveness,
            ligand_add_h,
            receptor_add_h,
            remove_nonstd_resi,
            qvina_seed,
            cleanup_files
        ],
        outputs=[output_zip_file, results_table, error_popup]
    )


app.launch(
    favicon_path=TAB_FAVICON,
    share=opt.public,
    server_name=opt.server_address,
    server_port=opt.server_port
)
