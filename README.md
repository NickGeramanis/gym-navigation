# Navigation Environment for [Gymnasium](https://gymnasium.farama.org)

The navigation environment is a single-agent domain featuring discrete
action space and continuous state space. It is a simulation of the autonomous
navigation problem in robotics.

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

The environment provides two tasks:

### NavigationTrack

The goal of this environment is to navigate a robot on a track without crashing
into the walls. Initially, the robot is placed randomly into the track but at a
safe distance from the walls. The state space consists of 5 range measurements.
The action space consists of 3 actions (move_forward, rotate_left,
rotate_right). Furthermore, both actions and states have additive white
Gaussian noise. The robot is rewarded +5 for moving forward and -0.5 for
rotating. If the robot crashes into the wall it is penalized with -200.

![Navigation environment](/images/navigation_track.png)

### NavigationGoal

The goal of this environment is to navigate a robot to reach a goal location
while avoiding the obstacles in the track. Initially, the robot is placed
randomly into the track but at a safe distance from the obstacles. Both the
goal and the obstacles change location in every episode. The state-space
consists of 5 range measurements plus the distance and angle from the goal. The
action space consists of 3 actions (move_forward, rotate_left, rotate_right).
Furthermore, both actions and states have additive white Gaussian noise. The
robot is rewarded +200 for reaching the goal and -200 for crashing. Moreover,
after each time step, the robot is rewarded or penalized proportionally to the
difference between the current and previous step distances from the goal.

![NavigationGoal environment](/images/navigation_goal.png)

## Getting Started

### Prerequisites

The following libraries need to be installed:

- NumPy
- Pygame
- Gymnasium

### Installation

Install the package from the repository with the following commands:

```bash
git clone https://github.com/NickGeramanis/gym-navigation
cd gym-navigation
pip3 install -e .
```

## Usage

You can use these environments in order to test and compare different
reinforcement learning algorithms.

You can create an instance of the environment with the following commands:

```python
import gymnasium as gym

# NavigationTrack Environment
env = gym.make('gym_navigation:NavigationTrack-v0',
               render_mode='human',
               track_id=1)
```

```python
import gymnasium as gym

# NavigationGoal Environment
env = gym.make('gym_navigation:NavigationGoal-v0',
               render_mode='human',
               track_id=2)
```

Currently, only one track has been implemented in each environment.

Furthermore, some unit tests have been implemented in the folder `tests` to verify
the proper functioning of the code.

## Status

Under maintenance.

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

## Authors

[Nick Geramanis](https://www.linkedin.com/in/nikolaos-geramanis)

