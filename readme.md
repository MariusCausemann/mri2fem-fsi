### Getting started 

This folder contains the code of the chapter **The pulsating brain: an interface-coupled fluid-poroelastic interaction model of the cranial cavity**. To get started, please install the requirements specified in the `environment.yml` file in this folder, e.g. with ```conda env create -f environment.yml```.

The notebooks build upon each other and are meant to be run in the following order: `MeshGeneration.ipynb`, `RunSimulation.ipynb`, `PostProcessing.ipynb` and `Visualization.ipynb`.
`ImageBrainMesh.ipynb` contains the code to generate a mesh from image-derived surface files of an actual brain.

To see the K3D animation within jupyter-notbook, you will have to enable the K3D extension with
```
jupyter nbextension install --py --user k3d
jupyter nbextension enable --py --user k3d```.