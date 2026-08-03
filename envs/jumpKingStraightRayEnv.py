from envs.jumpKingBaseEnv import JumpKingBaseEnv
from envs.jumpKing100Env import JumpKing100Env
from envs.jumpKingPunishEnv import JumpKingPunishEnv
from agents.jumpKingAgentStraightRay import JumpKingAgentStraightRay



class JumpKingStraightRayEnv(JumpKing100Env):

    def __init__(self):

        super().__init__(
            JumpKingAgentStraightRay
        )