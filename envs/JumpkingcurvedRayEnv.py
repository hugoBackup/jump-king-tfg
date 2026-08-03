from envs.jumpKingBaseEnv import JumpKingBaseEnv
from envs.jumpKing100Env import JumpKing100Env
from envs.jumpKingPunishEnv import JumpKingPunishEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay


class JumpKingCurvedRayEnv(JumpKing100Env):

    def __init__(self):

        super().__init__(
            jumpKingAgentCurvedRay
        )