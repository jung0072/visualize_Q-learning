import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 400}

    def __init__(self, render_mode=None, size=(12,4)):
        self.xsize, self.ysize = size
        self.window_xsize = 3*128
        self.window_ysize = 128
        self._cliff = False

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1,self.ysize-1]), shape=(2,), dtype=int)
,
                "target": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1,self.ysize-1]), shape=(2,), dtype=int)
,
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        
        # Location of cliff
        self._cliff_locations = [
            np.array([1,3]),
            np.array([2,3]),
            np.array([3,3]),
            np.array([4,3]),
            np.array([5,3]),
            np.array([6,3]),
            np.array([7,3]),
            np.array([8,3]),
            np.array([9,3]),
            np.array([10,3]),
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Agent always starts at the bottom left corner
        self._agent_location = np.array([0,3])

        # Target location will always be the right bottom corner
        self._target_location = np.array([11,3])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        min_values = np.array([0, 0])
        max_values = np.array([11, 3])
        #print("original location: ", self._agent_location)
        #print("direction: ", direction)
        #print("new location before clip: ", self._agent_location + direction)
        self._agent_location = np.clip(
	    self._agent_location + direction, min_values, max_values
	)
        #print("after action after clip: ", self._agent_location)
	
	# If the agent_location is on cliff set self.cliff True
        if any(np.array_equal(self._agent_location, cliff) for cliff in self._cliff_locations):
            self._cliff = True
            self._agent_location = np.clip(
	    self._agent_location - direction, min_values, max_values
	)

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        if terminated:
            reward = 1
            # print("reached to the destination")
        elif self._cliff:
            #print("Fall cliff")
            reward = -1
            self._cliff = False
        else:
            reward = 0
            
        observation = self._get_obs()
        info = self._get_info()
        #print("observation: ", observation["agent"])
        #print("-----------------------------------")

        if self.render_mode == "human":
            self._render_frame()
        
        
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_xsize, self.window_ysize))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_xsize, self.window_ysize))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_xsize / self.xsize
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.ysize + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_ysize, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_ysize),
                width=3,
            )
        for x in range(self.xsize + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_xsize, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_xsize),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
