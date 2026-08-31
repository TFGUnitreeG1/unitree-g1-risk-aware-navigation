# Unitree G1 Risk-Aware Navigation

Locomotion, waypoint navigation, visual perception and risk-aware mobility for the Unitree G1 humanoid robot in simulated search-and-rescue environments.

This repository contains the main code, trained locomotion checkpoints, simulation scenarios, waypoint routes and post-processing tools used to develop and evaluate a navigation framework for the Unitree G1 humanoid robot.

The framework combines reinforcement-learning-based locomotion with autonomous waypoint navigation, RGB-D perception, YOLO-based terrain detection and a spatial risk map used to modify the planned route.

## Overview

The system is structured around four main components:

- **Locomotion:** reinforcement-learning policy for the Unitree G1.
- **Navigation:** autonomous waypoint tracking and state-based navigation logic.
- **Perception:** RGB-D processing and YOLO-based terrain classification.
- **Risk-aware mobility:** adaptation of waypoint routes according to the estimated traversability risk.

Three mobility strategies are implemented:

- `uninformed`: the predefined waypoint route is followed without using the risk map to modify it.
- `semi_informed`: route adjustments obtained from previous navigation failures can be reused in subsequent executions.
- `informed`: the risk map is consulted during the mission and waypoints can be modified online according to the estimated local risk.

## Repository structure

unitree-g1-risk-aware-navigation/
│
├── locomotion/
│   ├── checkpoints/
│   │   ├── model_9400.pt
│   │   ├── model_44700.pt
│   │   └── model_49700.pt
│   │
│   └── configs/
│       ├── rsl_rl_ppo_cfg.py
│       └── velocity_env_cfg.py
│
├── navigation/
│   └── play_nave_mapa_semantico_informado.py
│
├── perception/
│   ├── g1_inference.py
│   └── g1_inference_online.py
│
├── scenarios/
│   ├── nave_densidad_alta.usd
│   ├── nave_densidad_alta_risk.usd
│   ├── nave_densidad_alta_sin_friccion.usd
│   ├── nave_densidad_baja.usd
│   ├── nave_densidad_media.usd
│   └── nave_simreal.usd
│
├── waypoints/
│   ├── waypoints_densidad_risk.csv
│   ├── waypoints_equivalente_real.csv
│   ├── waypoints_normal.csv
│   └── waypoints_zigzag.csv
│
├── evaluation/
│   ├── analysis/
│   │   ├── generar_resumen_campana.py
│   │   └── postprocesar_campana.py
│   │
│   └── config/
│       └── matriz_pruebas_congelada.csv
│
├── .gitignore
└── README.md


## Locomotion

The `locomotion/` directory contains the configuration files associated with the reinforcement-learning locomotion task and the checkpoints retained from the training process.

### Configuration

`velocity_env_cfg.py` contains the environment configuration used for the Unitree G1 locomotion task, including observations, commands, rewards, terrain configuration and termination conditions.

`rsl_rl_ppo_cfg.py` contains the PPO configuration used with RSL-RL.

### Checkpoints

Three checkpoints are retained:

| Checkpoint | Purpose |
| --- | --- |
| `model_9400.pt` | Intermediate policy used to evaluate the evolution of locomotion performance |
| `model_44700.pt` | Late training-stage checkpoint |
| `model_49700.pt` | Final locomotion policy used in the main experimental campaign |

The checkpoints are intended to be loaded through the RSL-RL integration provided by Unitree RL Lab.

## Autonomous navigation

The main navigation implementation is:

navigation/play_nave_mapa_semantico_informado.py


The script integrates the trained locomotion policy with autonomous waypoint navigation.

Velocity commands are generated from the relative position and orientation of the robot with respect to the active waypoint. The navigation logic also monitors events related to obstacle proximity, lack of progress and robot stability.

The navigation system includes the following states:

NORMAL
OBSTACULO
STOPPED
FALL_RISK
BLOCKED


The script supports the three mobility modes through:

--mobility-mode uninformed
--mobility-mode semi_informed
--mobility-mode informed


The main experiment-specific arguments are:

--scene
--waypoints-file
--test-id
--output-dir
--rep

Additional RSL-RL and Isaac Lab arguments are provided through the Unitree RL Lab execution environment.

During each mission, the script records robot pose, waypoint progress, velocity commands, measured velocities, orientation, contact forces, navigation states and mission events in a CSV log.

RGB and depth information are also recorded periodically for the risk-map pipeline.

## Perception and risk map

The perception system is located in:

perception/
├── g1_inference.py
└── g1_inference_online.py


### `g1_inference.py`

This module performs YOLO inference using RGB images together with depth information.

For each valid detection, the system estimates its position relative to the robot and assigns a risk value according to the detected terrain class.

The detections are projected onto a two-dimensional spatial grid to progressively build a semantic risk map.

The module also stores information such as:

- detected class;
- confidence;
- estimated depth;
- estimated spatial position;
- assigned risk;
- whether the detection lies inside the navigation region of interest.

### `g1_inference_online.py`

This script implements the online processing loop used during a simulation mission.

It monitors the active mission directory, reads newly generated RGB-D frames and robot pose information, processes each frame through `g1_inference.py` and progressively updates the risk map.

This allows the navigation process and the perception process to operate separately while exchanging information through the mission output files.

## Risk-aware mobility

The generated risk map can be used to modify the original waypoint route.

### Uninformed mobility

The original waypoint sequence is preserved throughout the mission. Perception information may be recorded, but it is not used to alter the navigation route.

### Semi-informed mobility

Route modifications associated with previous navigation problems can be stored and reused in later executions.

This provides a simple form of accumulated route knowledge without continuously replanning every waypoint from the current map.

### Informed mobility

The risk map is periodically evaluated during navigation.

If the local risk associated with a waypoint exceeds the defined criterion, alternative nearby positions are evaluated and the waypoint can be displaced toward a lower-risk region.

The original and modified routes are stored separately so that the effect of the adaptation can subsequently be evaluated.

## Simulation scenarios

The `scenarios/` directory contains the USD environments used in the experimental campaign.

| Scenario | Use |
| --- | --- |
| `nave_densidad_baja.usd` | Low obstacle-density environment |
| `nave_densidad_media.usd` | Medium obstacle-density environment |
| `nave_densidad_alta.usd` | High obstacle-density environment |
| `nave_densidad_alta_risk.usd` | High-density environment used for risk-aware experiments |
| `nave_densidad_alta_sin_friccion.usd` | Modified-friction scenario used to evaluate locomotion robustness |
| `nave_simreal.usd` | Scenario used for the simulation/physical-robot qualitative comparison |

Only the final scenario versions used in the experimental methodology are included.

## Waypoint routes

The predefined navigation routes are stored as CSV files in `waypoints/`.

waypoints_densidad_risk.csv
waypoints_equivalente_real.csv
waypoints_normal.csv
waypoints_zigzag.csv


Each file defines the planar coordinates of the waypoints used for a particular experiment.

During risk-aware navigation, additional copies of the original and current waypoint sets can be generated for each execution in order to preserve the initial route and record its modifications.

## Experimental evaluation

The repository includes the post-processing tools used to obtain quantitative metrics from the simulation campaign.

evaluation/
├── analysis/
│   ├── generar_resumen_campana.py
│   └── postprocesar_campana.py
└── config/
    └── matriz_pruebas_congelada.csv


### `postprocesar_campana.py`

Processes an individual experimental run and calculates metrics related to:

- mission completion and termination condition;
- waypoint progress;
- waypoint arrival times;
- minimum waypoint error;
- theoretical and actual travelled distance;
- linear and angular velocity;
- roll and pitch stability;
- foot contact forces;
- blocking and fall-risk events;
- perception statistics;
- risk-map coverage;
- waypoint modifications;
- risk reduction after route adaptation;
- lateral locomotion;
- turning behaviour.

The processed results are stored in CSV files for subsequent analysis.

### `generar_resumen_campana.py`

Processes the complete set of experimental executions and generates a consolidated summary.

The experimental matrix:

evaluation/config/matriz_pruebas_congelada.csv


contains the configuration and classification of the different runs used in the campaign.

## Experimental campaign

The included resources were used to evaluate the system under several conditions:

1. Evolution of the reinforcement-learning locomotion policy.
2. Navigation under different obstacle-density levels.
3. Comparison between normal and zigzag waypoint routes.
4. Locomotion under modified friction conditions.
5. Comparison between uninformed, semi-informed and informed mobility.
6. Qualitative comparison between simulated manoeuvres and the physical Unitree G1 platform.

The physical-robot experiments use the locomotion capabilities available on the robot and are intended as a qualitative comparison with equivalent simulated manoeuvres rather than as a direct deployment of the trained simulation policy.

## Software environment

The project was developed using:

- NVIDIA Isaac Sim
- Isaac Lab
- Unitree RL Lab
- RSL-RL
- Python
- PyTorch
- Gymnasium
- NumPy
- OpenCV
- Ultralytics YOLO
- Pandas
- Matplotlib

The simulation and reinforcement-learning components require a compatible Isaac Lab and Unitree RL Lab installation.

## External resources

This repository does not include:

- Isaac Sim;
- Isaac Lab;
- Unitree RL Lab;
- the YOLO model weights;
- complete reinforcement-learning training logs;
- raw data from the full experimental campaign;
- temporary or discarded development files.

The YOLO weights must therefore be supplied separately.

## Execution notes

This repository preserves the main files used during development and evaluation, but it is not distributed as a standalone Python package.

The navigation and locomotion scripts are intended to be integrated into a compatible Unitree RL Lab / Isaac Lab workspace.

Experiment-specific resources such as simulation scenarios, waypoint routes and output directories can be provided through the corresponding command-line arguments.

The YOLO model used by the perception pipeline is not included in the repository and must be supplied explicitly:

```text
--model /path/to/model.pt
```

The online perception process uses:

```text
outputs/datos_mapa_semantico/
```

as the relative directory for exchanging mission information with the navigation process. The navigation and online perception processes should therefore be launched from the same working directory.

For campaign post-processing, the root directory containing the experimental runs is supplied through:

```text
--root /path/to/experimental/runs
```

while the default experimental matrix is obtained directly from:

```text
evaluation/config/matriz_pruebas_congelada.csv
```

## Reproducibility

The repository provides the main configuration, navigation, perception and evaluation files required to reconstruct the experimental methodology.

Paths associated with external resources and experimental data are provided through command-line arguments or relative repository paths, avoiding dependencies on the original development machine.

Exact reproduction additionally depends on the external simulator and reinforcement-learning environment, the trained YOLO weights and the software versions used during development.

## Third-party software

Parts of the project build upon external open-source software, particularly Isaac Lab, Unitree RL Lab and RSL-RL.

Files derived from third-party projects retain their corresponding copyright and license headers where applicable.

The terms of the original projects should be reviewed before redistributing or reusing those components.

