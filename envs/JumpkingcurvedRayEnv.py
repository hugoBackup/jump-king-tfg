from envs.jumpKingBaseEnv import JumpKingBaseEnv
from envs.jumpKingPunishEnv import JumpKingPunishEnv
from envs.jumpKingKissEnv import JumpKingKissEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay


class JumpKingCurvedRayEnv(JumpKingKissEnv):

    def __init__(self):

        super().__init__(
            jumpKingAgentCurvedRay
        )