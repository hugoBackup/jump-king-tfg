from envs.jumpKingBaseEnv import JumpKingBaseEnv

from envs.jumpKingPunishEnv import JumpKingPunishEnv
from envs.jumpKingKissEnv import JumpKingKissEnv
from agents.jumpKingAgentStraightRay import JumpKingAgentStraightRay


#esta funcion sirve para poder ejecutar el espacio de observacion StraightRay con la recompensa deseada
class JumpKingStraightRayEnv(JumpKingPunishEnv):

    def __init__(self):

        super().__init__(
            JumpKingAgentStraightRay
        )