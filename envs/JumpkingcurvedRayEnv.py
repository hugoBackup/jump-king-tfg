from envs.jumpKingBaseEnv import JumpKingBaseEnv
from envs.jumpKingPunishEnv import JumpKingPunishEnv
from envs.jumpKingKissEnv import JumpKingKissEnv
from agents.jumpKingAgentCurvedRay import jumpKingAgentCurvedRay

#esta funcion sirve para poder ejecutar el espacio de observacion CurvedRay con la recompensa deseada
class JumpKingCurvedRayEnv(JumpKingPunishEnv):

    def __init__(self):

        super().__init__(
            jumpKingAgentCurvedRay
        )