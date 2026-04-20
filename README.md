# **LCGNav: Local Candidate-Aware Geometric Enhancement for General Topological Planning in Vision-Language Navigation**

![Figure 1](assets\Figure 1.png)

# Abstract

Online topological planning has advanced Vision-Language Navigation in Continuous Environments (VLN-CE) by abstracting complex environments into structured graph representations. However, existing methods still face two key bottlenecks. First, as the global topological map expands, the accumulation of historical visited nodes may dilute the cross-modal planner’s focus on local frontier candidates (ghost nodes), weakening the model’s perception of current exploration directions. Second, standard approaches typically rely on 2D depth maps for local spatial awareness. Since the agent’s navigable distance per step is inherently limited, distant regions in depth maps often introduce spatial redundancy, which can hinder the extraction of effective local features. To address these issues, we propose LCGNav, a modular Local Candidate-Aware Geometric Enhancement framework with strong cross-architecture transferability. LCGNav explicitly converts depth maps into 3D point clouds and applies physical truncation based on the agent’s reachable distance, thereby filtering out distant spatial redundancy and extracting more informative local geometric features. To mitigate this issue, we introduce a Dimension-Preserving Local Focus Fusion mechanism coupled with a transient state degradation strategy. It injects these geometric features exclusively into the currently connected ghost nodes without altering the original topological feature dimensions. Once a ghost node is visited, its enhanced representation is reset to the standard topological feature, which helps reduce the influence of historical trajectories and encourages the planner to focus on the current action space. Experiments on the R2R-CE dataset show that LCGNav serves as a robust cross-architecture enhancement module, improving one or more key metrics of representative baselines (e.g., ETPNav, DGNav, and BEVBert) with low additional training overhead. Notably, when integrated with the ETP-R1 baseline, our framework achieves the best performance among the compared online topological methods on the R2R-CE val-unseen split. 

### Installation

1. This project is developed with Python 3.7. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:

```bash
conda create -n vlnce python=3.7
conda activate vlnce
```

2. Install [habitat-sim](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.7_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2) with the corresponding Python version and headless mode:

```bash
conda install habitat-sim-0.1.7-py3.7_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
```

3. Then install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7):

​	**Notice:** You need to comment out the TensorFlow line in `habitat_baselines/rl/requirements.txt`.

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

4. Clone this repository and install all requirements for `habitat-lab`, VLN-CE and our experiments. Note that we specify `gym==0.21.0` because its latest version is not compatible with `habitat-lab-v0.1.7`.

```bash
git clone git@github.com:shannanshouyin/LCGNav.git
cd LCGNav
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install gym==0.21.0
python -m pip install -r requirements.txt
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html # Only for BEVbert
```

### Scenes: Matterport3D

Instructions copied from [VLN-CE](https://github.com/jacobkrantz/VLN-CE):

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract such that it has the form `scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes. Place the `scene_datasets` folder in `data/`.


## Running

1. The pre-training method and weights are inherited from  [ETPNav](https://github.com/MarSaKi/ETPNav),  [BEVBert](https://github.com/MarSaKi/VLN-BEVBert), [DGNav](https://github.com/shannanshouyin/DGNav), and  [ETP-R1](https://github.com/Cepillar/ETP-R1). Among them, ETPNav, BEVBert, and DGNav share the same pre-trained weights, while ETP-R1 uses enhanced pre-trained weights from a larger model.

2. Post-training is performed based on the original fine-tuned weights from ETPNav, BEVBert, DGNav, and ETP-R1.

​	Use `main.bash` for `Training/Evaluation/Inference with a single GPU or with multiple GPUs on a single node.` Simply adjust the arguments of the bash scripts.

The running commands for ETPNav, BEVBert, and DGNav are as follows:

```
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash train 2333  # training
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash eval  2333  # evaluation
```

The running commands for ETP-R1 are as follows:

```
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash dagger 2333  # training
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash eval  2333  # evaluation
```

## Acknowledge

Our implementations are partially inspired by  [ETPNav](https://github.com/MarSaKi/ETPNav), [BEVBert](https://github.com/MarSaKi/VLN-BEVBert), [ETP-R1](https://github.com/Cepillar/ETP-R1).

Thanks for their great works!

## Performance Demonstration

![Table](assets\Table.png)

![Figure 4](assets\Figure 4.png)
