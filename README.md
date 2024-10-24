
[![Docs](https://img.shields.io/badge/doc-pdf-red)](Documentacion.pdf)
[![Dataset](https://img.shields.io/badge/dataset-download-brightgreen)](https://drive.google.com/drive/folders/121s67_IgW-r39hG-xwHIUI32-_QIE97X?usp=sharing)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Ignacio%20Boero-blue?style=flat&logo=google-scholar)](https://scholar.google.com/citations?user=abc123)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Santiago%20Díaz-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sdiazvaz/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tomás%20Vázquez-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/tom%C3%A1s-v%C3%A1zquez-292623186/)
[![Facultad de Ingeniería](https://img.shields.io/badge/Facultad%20de%20Ingeniería-UdelaR-blue)](https://www.fing.edu.uy)


# DORAA-UY - Optimal Reactive Power Dispatch using Machine Learning

This repository is part of the thesis project developed at the Facultad de Ingeniería of UdelaR. The main objective is to demonstrate how machine learning techniques can be applied to efficiently solve the Optimal Reactive Power Dispatch (ORPD) problem in the Uruguayan electrical grid. In recent years, the exploration of these techniques for decision-making on control variables in electrical networks has been a focus of study by the community, making it a hot topic today. This thesis tackles the problem using supervised and unsupervised learning techniques, and neural network architectures such as FCNN (Fully Connected Neural Networks) and GNN (Graph Neural Networks). This work builds upon the efforts of Damian Owerko, Fernando Gama, and Alejandro Ribeiro ([link1](https://arxiv.org/abs/1910.09658), [link2](https://arxiv.org/abs/2210.09277)).

The project aims to optimize the use of network resources, improving both its stability and energy efficiency. As a starting point, the IEEE 30 and IEEE 118 test grids are used, which are later extended to the Uruguayan grid model with real data provided by the National Load Dispatch (DNC). It should be noted that active power dispatch in the Uruguayan grid is relatively simple due to the high penetration of renewable energies. Therefore, this thesis assumes that active power dispatch is known, focusing primarily on optimal reactive power generation.

This repository includes the code necessary to train machine learning models that optimize ORPD for each of the grids. The database and the model of the Uruguayan grid are available for download [here](https://drive.google.com/drive/folders/121s67_IgW-r39hG-xwHIUI32-_QIE97X?usp=sharing).

Below is a detailed description of the project. All concepts presented in this repository are thoroughly addressed in the [thesis documentation](Documentacion.pdf), so the descriptions in the following sections are just an introduction to the problem. For in-depth study, it is highly recommended to refer to the documentation. For the moment, the documentation is writen in spanish, but a paper in english is coming soon. If you have specific questions, feel free to open an issue in this repository or contact us. Additionally, a detailed explanation of how to execute the scripts is provided [below](#uso-del-código). 

---

# Optimal Reactive Power Dispatch in the Uruguayan Electrical Grid using Machine Learning

This project focuses on the application of machine learning strategies to solve the Optimal Reactive Power Dispatch (ORPD) problem in the Uruguayan electrical grid, with the goal of minimizing losses and ensuring system stability. To this end, neural networks, both fully connected (FCNN) and graph neural networks (GNN), are used, leveraging the latter's ability to model the electrical grid structure as a graph.

The problem is approached through supervised and unsupervised learning. The code primarily uses the PandaPower library, an open-source tool developed in Python for electrical power system analysis. Its capabilities include load flow and optimal power flow (OPF) analysis, with PyPower as its optimizer. Additionally, it allows for reactive power dispatch optimization (ORPD), which is critical to this research. PandaPower integrates with PowerModels, an optimizer developed in Julia, so when running optimization, Julia is invoked to execute the optimization algorithm.

Moreover, the project includes studies on IEEE30 and IEEE118 test grids to compare the strategies with more general cases and analyze the representativeness of synthetic data against real Uruguayan grid data. One of the main contributions of this work is the creation of a real historical database of the Uruguayan electrical grid, which will be made publicly available to foster further research in this area.

## Optimization Problem Formulation

The Optimal Reactive Power Dispatch (ORPD) problem is framed as an optimization problem subject to constraints. Several formulations of this problem exist, but the main objective is to minimize electrical losses in the grid while respecting its operational and security limits. The problem constraints represent the physical limitations imposed by the grid components on the values that the variables can take. The first constraint is the power flow equation, which establishes the relationship between voltages and injected powers throughout the grid; thus, any proposed solution must satisfy it. Additionally, there are bus-level constraints, limiting both voltage magnitudes and angles. There is also a physical limit on the current that a line or transformer can continuously handle, which must be met for current flowing in both directions. In terms of powers, there are constraints on the capacity to generate reactive power for voltage-controlling generators and reactive power compensators. Finally, there are constraints on the active and reactive power of the reference generator.


## IEEE Grids
The IEEE power grids are a set of transmission networks presented as publicly available test cases by the University of Illinois in the early 1990s. These consist of seven grids ranging in size from 14 to 300 buses. These networks provide a controlled and well-documented environment for testing and evaluating optimization algorithms and techniques before their implementation in the Uruguayan grid. They also allow exploration of different architectures and strategies, facilitating the identification of best practices and potential challenges that may arise when applied to the Uruguayan grid. Due to the popularity of these grids, the PandaPower library already includes an implementation for them, making their use in electrical network simulations and analysis even easier.

In this work, two of these grids are used: IEEE30 and IEEE118, which have 30 and 118 buses, respectively. The reason for using IEEE30 is that it is a small-sized grid, which, due to its size, makes the results more intuitively interpretable, which is very useful when identifying potential error causes. On the other hand, the IEEE118 grid is chosen because its number of buses is similar to that of the Uruguayan grid, which has 107 buses. Thus, an operational strategy for the 30-bus grid can be assumed to be valid for a grid of similar size to the Uruguayan one.

## Uruguayan Electrical Grid

The grid model is built in PandaPower, and its file can be found in the repository and loaded via:
`net = pp.from_pickle('uru_net.p')`

This model corresponds to a simplified version of the Uruguayan electrical grid, which has a total of 107 buses, of which 95 are at 150 kV and 12 at 500 kV. There are also 144 lines, 14 of which connect 500 kV buses, and 130 connect 150 kV buses. All 500 kV buses have associated transformers that connect them to a bus of the same name at 150 kV, so there are also 12 transformers. Additionally, there are 43 generators, of which 15 are voltage controllers, 27 are static, and there is 1 reference generator. Finally, there are 55 loads and 6 reactive power compensators.

## Input and Output Definition
The main objective is, given a state of the grid (i.e., knowing the active and reactive power of the loads and the active power of the generators), to determine the setpoint voltage for the generators in the grid and the reactive power values to dispatch at the compensators. When configuring the generator voltage, and with the active power they generate as input, the generators will generate the reactive power needed to maintain that voltage level at the bus to which they are connected, making voltage control directly impact the reactive power in the grid.

## Problem-solving Approaches

In this work, two different methodologies are applied to solve this problem. First, we explore supervised learning algorithms. For this, a database of different grid states along with their respective optimal control variables is used, previously found through PandaPower’s solver. Thus, FCNN and GNN models are trained to learn this input-output mapping. These models are expected to find the same optimal solutions as a traditional optimizer, resulting in similar or worse performance in terms of grid losses. Nevertheless, training a model has the advantage of offline computation, significantly reducing inference times. Mean Squared Error (MSE) is used as the loss function.

In addition, as an exploration of different results, an unsupervised learning model is implemented, where the loss function is a variant of the optimization problem's Lagrangian. In this case, the model does not try to replicate optimizer outputs but iteratively learns to minimize grid losses while respecting the problem constraints.

## Data
- IEEE
  
  Regarding the IEEE networks, neither offers historical generation or power demand data to train the models. This means that for working with these networks, it's necessary to generate synthetic data. The data required to simulate are those used as inputs to the problem. Active and reactive power demand values, as well as active power generation values, need to be generated (static generators are omitted since this network does not include them).

  The methodology used for synthetic data generation follows a similar process to other works addressing this problem using machine learning. For each node, nominal values of active and reactive power from all nodes are taken. These nominal values are provided by the network. Then, based on these values, a distribution of generation/demand is generated, which is uniformly distributed between 0.7 and 1.3 of the reference value. Using these values, the optimum is found using the `net.acopf()` function (I don’t remember the exact name), and the optimal voltage values for the generators are recorded. These values will be the labels for training the supervised learning models.


- Uruguayan Power Grid

  For the Uruguayan power grid, historical data is available from January 2021, with records at one-hour intervals. These were provided by the National Load Dispatch (DNC) and correspond to active power generation values from the generators and active power demand values. Regarding reactive power, this data is unavailable (both for generation and demand), so it is generated synthetically. Reactive values are sampled by taking the active power of the loads multiplied by the cosine of an angle that is set to 0.995 for nighttime data (between 00:00 and 06:00) and 0.980 for the rest of the day. For static generators, reactive power values are set to 0, as it is assumed they only generate active power.

## Implementation Details

The two architectures tested in this work are fully connected neural networks and graph-based neural networks. These consist of a sequence of linear or convolutional layers, interleaved with activation functions. Leaky ReLU is used as the activation function.

In addition to the linear layers and nonlinearity, a batch normalization layer is added. This layer normalizes the output of the hidden layers of the neural network, which helps stabilize and accelerate the training process. The use of these layers is optional and treated as a training hyperparameter.

Another interesting implementation detail is the use of a mask on the predictor's output. The optimal output consists of voltages for controllable generators and reactive power for compensators for all buses, with zeros for buses not connected to these elements. Therefore, using a mask to multiply the predictor's output, setting to zero the outputs that are always null, prevents the model from wasting capacity learning to drive these inputs to zero, potentially improving performance.

The loss function used for supervised learning training is MSE, as the goal of the models is to replicate the target values. On the other hand, as previously mentioned, a modification of the Lagrangian from the optimization problem is used for training the unsupervised learning models.

## Hyperparameter Search

One strategy used in this work is leveraging the [optuna](https://optuna.org) tool for intelligent hyperparameter search. Various aspects are explored, including different architecture sizes, batch sizes, learning rates, the use of batch normalization, and normalization of input and output data, among others.

---

# Code usage

## Python Environment Setup
- Install Julia with the PyCall tool. Follow the instructions at this [link])(https://pandapower.readthedocs.io/en/v2.6.0/opf/powermodels.html)
- Create a Python environment and install all dependencies using the following command:

` pip install requirements.txt`

## Data Generation
As mentioned earlier, this repository contains scripts for data generation, both for IEEE networks and the Uruguayan grid.

- IEEE

  To generate data for the IEEE networks, run the script `generar_datos.py` located in the folder `supervisado/IEEE/data/`. You need to specify the network for which you want to generate the data, as well as the amount of data to be generated. In the example below, data for the IEEE30 network is generated with a quantity of 1000 samples. Note that in this case, there’s no need to download the IEEE networks beforehand, as they are included in the PandaPower library.

  ```
  python generar_datos.py --red 30 --N 1000
  ```

- Uruguayan Grid

  For the Uruguayan power grid, synthetic data can also be generated, which is very useful before working with real data. To generate this data, run the script `generar_datos_sintetica.py` located in the folder `supervisado/URU/data/`. In this case, there’s no need to specify the network. Below is an example where 1000 data samples are generated. The pickle file with the network model `red_uru.p` must be located in the same folder. This model can also be downloaded from Drive if necessary.

  ```
  python generar_datos_sintetica.py --N 1000
  ```


## Model Training
To train the models, the necessary data must first be placed in the correct folders. These can be downloaded from the previously mentioned Drive link. To avoid errors when running the training scripts, ensure that the data is structured as follows (for example, in the case of training supervised models with the Uruguayan grid):

```
/supervisado
│
└───/URU
    │
    ├───/data
    │   └───/reduru
    │       ├───/train
    │       ├───/test
    │       └───/val
    │
    ├───/entrenamiento
    │   └───(scripts related to model training)
    │
    └───/resultados
        └───(reports, metrics, result graphs, etc.)

```

Additionally, besides the data, you must have a configuration file, which can be found in the `configs/` folder of each training type. Some examples are provided in this repository. If you wish to train a fully connected neural network, select the config files with the name FCNN, whereas if you want to train graph-based models, choose the files with the GNN abbreviation. You can also distinguish which power grid the files correspond to by the file names. Once all these requirements are met, the corresponding model can be trained by running the script `train.py` located in the respective folder. Note that when executing this code, a hyperparameter search is performed using Optuna. To configure this hyperparameter search, directly edit the `train.py` script with the parameters you wish to explore. To run the set of trainings, execute the following code:

```train.py --cfg <path-to-config.yaml>```

It is highly recommended to have GPU availability for training.

## Result Analysis
Lastly, within each folder, notebooks can be found for result analysis. It’s important to note that to run this notebook, you must have trained models available in the `runs/`. folder. Furthermore, this folder is organized by each power grid, and then by each model (GNN or FCNN). Therefore, when running the analysis code (`analisis_resultados.ipynb`), you should select in the first two lines of the second cell which model you want to evaluate. An important detail is that this notebook loads the model located in the `best`folder, which corresponds to the best obtained model. This is why the best model must be previously renamed as `best`. Upon cloning this repository, you will be able to see the best models obtained during this thesis.

This notebook covers various types of analysis for the best model, using the test data. First, it shows both the value of the metric obtained for this model and graphs of the predicted voltages for the generators compared to the target values, ordered from lowest to highest (this applies only to the supervised learning case, where the optimizer’s behavior is tracked). Then, an analysis is made of how good the model was as a solution to the ORPD. For this, comparison histograms are created for the costs, sample by sample, between the model and the optimizer, or against baseline models, calculating the ratio between the costs. Finally, an analysis is made of how feasible the solutions are, based on the percentage by which the network constraints were violated, as well as an inference time analysis compared to the optimizer.

## To Cite Our Work
To use our work, please use the following citation in BibTeX format:
```
@misc{GNN4OPFuru2024,
  author = {Boero, Ignacio and Diaz, Santiago and Vazquez, Tomas},
  title = {DORAA-UY},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/tomyvazquez/DORAA-UY},
  note = {GitHub repository},
}
```

## License

This project is licensed under the terms of the [Licencia MIT](LICENSE).


