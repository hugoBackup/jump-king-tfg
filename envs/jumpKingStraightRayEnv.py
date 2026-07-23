from envs.jumpKingBaseEnv import JumpKingBaseEnv
from agents.jumpKingAgentStraightRay import JumpKingAgentStraightRay


class JumpKingStraightRayEnv(JumpKingBaseEnv):

    def __init__(self):

        super().__init__(
            JumpKingAgentStraightRay
        )