
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from agents.jumpKingTreeSearchAgentPlus import jumpKingTreeSearchAgentPlus

os.environ["render"] = "1"

from JumpKing import JKGame



game = JKGame()

game.reset()


agent = jumpKingTreeSearchAgentPlus(game)

game.agent = agent

while True:

    action = None

    if game.move_available():

        action = agent.get_action()

        height = game.get_global_height(
            game.king.levels.current_level,
            game.king.y
        )

        print(
            "Acción:",
            action,
            "| Altura global antes del salto:",
            height
        )

    game.step(action)