"""This module contains the navigation class."""

from abc import abstractmethod
from typing import Tuple, Union, Optional

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

    metadata = {'render_modes': ['human']}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
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
    def _do_update_observation(self) -> None:
        pass

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None) -> Union[
            np.ndarray, Tuple[np.ndarray, dict]]:
        super().reset(seed=seed)
        self._do_init_environment()
        self._do_update_observation()
        return (self._observation, {}) if return_info else self._observation

    @abstractmethod
    def _do_init_environment(self) -> None:
        pass

    def render(self, mode: str = "human") -> None:
        if mode not in self.metadata['render_modes']:
            raise ValueError(f'Mode {mode} is not supported')

        plt.clf()
        self._do_plot()
        self._fork_plot()

        plt.pause(self._RENDER_PAUSE_TIME)

    @abstractmethod
    def _do_plot(self) -> None:
        pass

    def _fork_plot(self) -> None:
        pass

    def close(self) -> None:
        plt.close()
