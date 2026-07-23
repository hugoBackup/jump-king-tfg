from envs.jumpKingBaseEnv import JumpKingBaseEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay


class JumpKingCurvedRayEnv(JumpKingBaseEnv):

    def __init__(self):

        super().__init__(
            jumpKingAgentCurvedRay
        )