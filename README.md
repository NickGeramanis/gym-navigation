# Navigation Environment for OpenAI Gym

The navigation environment is a single-agent domain featuring discrete action-space and continuous state-space. It is a simulation of the autonomous navigation problem in robotics.

## Table of Contents


- [Description](#description)
  - [Navigation](#navigation)
  - [NavigationGoal](#navigationgoal)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [License](#license)
- [Authors](#authors)


## Description

Two tasks have been implemented:

### Navigation

The goal of this environment is to navigate a robot on a track without crashing into the walls. Initially, the robot is placed randomly into the track but at a safe distance from the walls. The state-space consists of 5 range measurements. The action-space consist of 3 action (move_forward, rotate_left, rotate_right). Furthermore, both actions and states have additive white Gaussian noise. The robot is rewarded +5 for moving forward and -0.5 for rotating. If the robot crashes into the wall it is penalized with -200.

![Navigation environment](/images/navigation_env.png)
### NavigationGoal

The goal of this environment is to navigate a robot to reach a goal location while avoiding the obstacles in the track. Initially, the robot is placed randomly into the track but at a safe distance from the obstacles. Both the goal and the obstacles change location in every episode. The state-space consists of 5 range measurements and the distance and angle from the goal. The action-space consist of 3 action (move_forward, rotate_left, rotate_right). Furthermore, both actions and states have additive white Gaussian noise. The robot is rewarded +200 for reaching the goal and -200 for crashing. Moreover, if the distance from the robot to its target position decreases after a time step, the robot is rewarded proportional to the reduced distance.

![NavigationGoal environment](/images/navigation_goal_env.png)


## Getting Started


### Prerequisites

The following libraries need to be installed:

- NumPy
- Matplotlib
- OpenAI Gym


### Installation

Install the package with:

```bash
pip3 install -e gym-navigation
```


## Usage

You can use these environments in order to test and compare different reinforcement learning algorithms.

You can create an instance of the environment with:

```python
gym.make('gym_navigation:Navigation-v0', track_id=1)
gym.make('gym_navigation:NavigationGoal-v0', track_id=1)
```

Currently, only one track has been implemented in each environment.

Furthermore, some unit tests have been implemented in folder `tests` to verify the proper functioning of the code.


## Status

Under maintenance.


## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.


## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

