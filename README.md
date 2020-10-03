# Navigation Environment for the OpenAI Gym

The Navigation environment is a single-agent domain featuring discrete action-space and continuous state space. It is a simple simulation of the autonomous navigation problem in robotics. Two tasks are supported:

## Navigation

The goal of this environment is to navigate a robot in different tracks without crashing into the walls. Currently, there are 3 available tracks (use track_id parameter). Initially, the robot is placed randomly into the track but at a safe distance from the walls. The state-space consists of 5 range measurements. The action-space consist of 3 action (move_forward, rotate_left, rotate_right). Furthermore, both actions and states have additive white Gaussian noise. The robot is rewarded +5 for moving forward and -0.5 for rotating. If the robot crashes into the wall it is penalized with -200.

## NavigationGoal

The goal of this environment is to navigate a robot to reach a goal location while avoiding the obstacles in the track. Initially, the robot is placed at (0, 0) and the goal at a random location. The state-space consists of 5 range measurements the distance and angle from the goal. The action-space consist of 3 action (move_forward, rotate_left, rotate_right). Furthermore, both actions and states have additive white Gaussian noise. The reward function consists of the environment reward, the direction reward, and the transition reward. Moreover, to encourage the robot to reach the target location as fast as possible the robot receives a penalty for every step.  


# Installation & Usage

Install the package with:

```bash
pip install -e gym-navigation
```

You can create an instance of the environment with

```python
gym.make('gym_navigation:Navigation-v0')
gym.make('gym_navigation:NavigationGoal-v0')
```