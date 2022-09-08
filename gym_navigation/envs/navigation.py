"""This module contains the navigation class."""

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
import pygame
from gym import Env
from gym.core import RenderFrame
from pygame import Surface
from pygame.time import Clock

from gym_navigation.enums.track import Track
from gym_navigation.geometry.line import Line


class Navigation(Env):
    """The navigation class.

    This class is used to define the step, reset, render, close methods
    as template methods. In this way we can create multiple environments that
    can inherit from one another and only redefine certain steps.
    """
    _WINDOW_SIZE = 700
    _RESOLUTION = 20  # 1m => 20 pixels
    _WIDTH = 3
    _X_OFFSET = 150
    _Y_OFFSET = -150

    _track: Track
    _world: Tuple[Line, ...]
    _window: Optional[pygame.surface.Surface] = None
    _clock: Optional[Clock] = None

    metadata: Dict[str, Any] = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 track_id: int = 1) -> None:
        if (render_mode is not None
                and render_mode not in self.metadata['render_modes']):
            raise ValueError(f'Mode {render_mode} is not supported')
        self.render_mode = render_mode  # type: ignore
        self._track = Track(track_id)
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._WINDOW_SIZE, self._WINDOW_SIZE))
            self._clock = Clock()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._do_perform_action(action)
        observation = self._do_get_observation()
        terminated = self._do_check_if_terminated()
        truncated = False
        reward = self._do_calculate_reward(action)
        info = self._do_create_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _do_perform_action(self, action: int) -> None:
        pass

    @abstractmethod
    def _do_get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def _do_check_if_terminated(self) -> bool:
        pass

    @abstractmethod
    def _do_calculate_reward(self, action: int) -> float:
        pass

    @abstractmethod
    def _do_create_info(self) -> dict:
        pass

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._world = deepcopy(self._track.walls)
        self._do_init_environment(options)
        observation = self._do_get_observation()
        info = self._do_create_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    @abstractmethod
    def _do_init_environment(self, options: Optional[dict] = None) -> None:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return None

    def _render_frame(self, mode: str = 'human') -> None:
        if self._window is None or self._clock is None:
            return

        canvas = Surface((self._WINDOW_SIZE, self._WINDOW_SIZE))
        self._do_draw(canvas)
        if mode == 'human':
            if self._window is None or self._clock is None:
                return
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])

    @abstractmethod
    def _do_draw(self, canvas: Surface) -> None:
        pass

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
