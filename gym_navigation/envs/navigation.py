"""This module contains the navigation class."""

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple, Union, Optional, List, Dict, Any

import numpy as np
import pygame
from gym import Env
from gym.core import RenderFrame
from gym.utils.renderer import Renderer
from pygame.surface import Surface
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

    _observation: np.ndarray
    _track: Track
    _world: Tuple[Line, ...]
    _renderer: Renderer
    _window: Optional[Surface]
    _clock: Optional[Clock]

    metadata: Dict[str, Any] = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 track_id: int = 1) -> None:
        if (render_mode is not None
                and render_mode not in self.metadata['render_modes']):
            raise ValueError(f'Mode {render_mode} is not supported')
        self.render_mode = render_mode
        self._track = Track(track_id)
        self._window = None
        if self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._WINDOW_SIZE, self._WINDOW_SIZE))
            self._clock = Clock()
        self._renderer = Renderer(self.render_mode, self._render_frame)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action} ({type(action)})')

        self._do_perform_action(action)
        self._do_update_observation()
        self._renderer.render_step()
        terminated = self._do_check_if_terminated()
        truncated = False
        reward = self._do_calculate_reward(action)

        return self._observation.copy(), reward, terminated, truncated, {}

    @abstractmethod
    def _do_perform_action(self, action: int) -> None:
        pass

    @abstractmethod
    def _do_check_if_terminated(self) -> bool:
        pass

    @abstractmethod
    def _do_calculate_reward(self, action: int) -> float:
        pass

    @abstractmethod
    def _do_update_observation(self) -> None:
        pass

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None
              ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        super().reset(seed=seed)
        self._world = deepcopy(self._track.walls)
        self._do_init_environment()
        self._do_update_observation()
        self._renderer.reset()
        self._renderer.render_step()
        return ((self._observation.copy(), {}) if return_info
                else self._observation.copy())

    @abstractmethod
    def _do_init_environment(self) -> None:
        pass

    def render(self,
               mode: str = 'human'
               ) -> Optional[List[RenderFrame]]:
        return self._renderer.get_renders()

    def _render_frame(self, mode: str = 'human') -> None:
        canvas = Surface((self._WINDOW_SIZE, self._WINDOW_SIZE))
        self._do_draw(canvas)
        self._fork_draw(canvas)
        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata["render_fps"])

    @abstractmethod
    def _do_draw(self, canvas: Surface) -> None:
        pass

    def _fork_draw(self, canvas: Surface) -> None:
        pass

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
