from time import sleep
import gymnasium
import pettingzoo
import pettingzoo.utils


class SB3Agent(pettingzoo.utils.BaseWrapper, gymnasium.Env):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space(self.env.agent_selection)
        self.action_space = self.env.action_space(self.env.agent_selection)

    def reset(self, seed=None, options=None):
        # Clears the board, resets game state, applies the seed, and returns the initial observation and info.
        super().reset(seed, options)

        # Get the observation for the first agent (assuming all agents have the same spaces)
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
        # Get the action space for the first agent (assuming all agents have the same spaces)
        self.action_space = super().action_space(self.possible_agents[0])

        """
        self.agent_selection => the current active player
        self.observe(agent) => returns the observation for the specified agent
        {} => empty info dictionary (you can add any relevant info here if needed)
        """
        return self.observe(self.agent_selection), {}

    """
    Returns the next observation, reward, done flag, and info dictionary after taking an action.

    The observation is for the next agent (used to determine the next action), while the remaining items are for the agent that just acted (used to understand what just happened).
    """

    def step(self, action):
        current_agent = self.agent_selection

        # Perform the action for the current agent and get the resulting observation, reward, done flag, and info
        super().step(action)

        next_agent = self.agent_selection

        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()
