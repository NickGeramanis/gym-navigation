"""This module contains the navigation class."""

from abc import abstractmethod
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym import Env


class Navigation(Env):
    """The navigation class.

    This class is used to define the step, reset, render, close, seed methods
    as template methods. In this way we can create multiple environments that
    can inherit from one another and only redefine certain steps.
    """
    _RENDER_PAUSE_TIME = 0.01
    _observation: np.ndarray

    def step(self,
             action: int) -> Tuple[np.ndarray, float, bool, Dict[str, str]]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._do_perform_action(action)
        done = self._do_check_if_done()
        reward = self._do_calculate_reward(action)
        # Update the observation last
        # so that we have access to previous observation.
        self._do_update_observation()

        return self._observation.copy(), reward, done, {}

    @abstractmethod
    def _do_perform_action(self, action: int) -> None:
        pass

    @abstractmethod
    def _do_check_if_done(self) -> bool:
        pass

    @abstractmethod
    def _do_calculate_reward(self, action: int) -> float:
        pass

    @abstractmethod
    def _do_update_observation(self):
        pass

    def reset(self) -> np.ndarray:
        self._do_init_environment()
        self._do_update_observation()
        return self._observation

    @abstractmethod
    def _do_init_environment(self) -> None:
        pass

    def render(self, mode="human"):
        if mode not in self.metadata['render.modes']:
            raise ValueError(f'Mode {mode} is not supported')

        plt.clf()
        self._do_plot()
        self._fork_plot()

        plt.pause(self._RENDER_PAUSE_TIME)

    @abstractmethod
    def _do_plot(self):
        pass

    def _fork_plot(self):
        pass

    def close(self):
        plt.close()
