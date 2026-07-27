
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from agents.jumpKingTreeSearchAgent import jumpKingTreeSearchAgent

os.environ["render"] = "1"
from envs.jumpKingTreeSearchEnv import JumpKingTreeSearchEnv
from JumpKing import JKGame



game = JKGame()

game.reset()

agent = jumpKingTreeSearchAgent(game)

game.agent = agent

while True:

    action = None

    if game.move_available():
        action = agent.get_action()

    game.step(action)