
# Create env
```bash
conda create -n geoldm python=3.10.13

conda activate geoldm
```
```bash
conda create -n geoldm-a100 python=3.10.13

conda activate geoldm-a100
```

</br>
</br>

# Install Deps
```bash
conda install -c conda-forge rdkit biopython openbabel

conda install pathtools==0.1.2 -y

pip install imageio numpy==1.23.3 scipy tqdm wandb==0.13.4 msgpack rdkit matplotlib==3.5.2 matplotlib-inline==0.1.6 chardet periodictable ipykernel jupyter notebook prettytable seaborn scikit-learn==1.5.1 gdown

pip install gradio==5.9 plotly==5.24 huggingface
```

</br>
</br>

# Install torch
For <code>geoldm</code>: titan, V100s GPUs (CUDA 11.8, sm_86):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For <code>geoldm-a100</code>: A100 GPUs (CUDA 12.1)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

</br>
</br>

# Update Qvina Script Permission
```bash
cd analysis/qvina
chmod +x qvina2.1
cd ../..
```

</br>
</br>

# Create env for mgltools 
This library is used to preprocess both ligand and protein pockets before performing docking analysis with <code>QuickVina2.1</code>. Specifically, it performs tasks like converting files from .pdb to .pdbqt (Adds change Q and torsions T to pockets), and etc.
```bash
conda create -n mgltools-python2 python=2.7 -y
conda activate mgltools-python2 
conda install -c bioconda mgltools -y
```

Example Usage:
```bash
conda activate mgltools-python2 
prepare_receptor4.py -h
```

or

```bash
import subprocess
subprocess.run('conda run -n mgltools-python2 prepare_receptor4.py -h', shell=True)
subprocess.run('conda run -n mgltools-python2 prepare_receptor4.py -r input_path.pdb -o output_path.pdbqt', shell=True)
```
