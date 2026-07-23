from envs.jumpKingBaseEnv import JumpKingBaseEnv
from agents.jumpKingTreeSearchAgent import jumpKingTreeSearchAgent

class JumpKingTreeSearchEnv(JumpKingBaseEnv):

    def __init__(self):

        super().__init__(
            jumpKingTreeSearchAgent
        )