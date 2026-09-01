
#funcion representacion de nodo para treesearch
class SearchNode:

    def __init__(
        self,
        x,
        y,
        level,
        global_height,
        action=None,
        jump_power=None,
        direction=None,
        parent=None
    ):

        self.x = x
        self.y = y
        self.level = level

        self.global_height = global_height

        # Acción (0-9) que llevó desde el padre hasta este nodo
        self.action = action

        # Potencia del salto utilizada
        self.jump_power = jump_power

        # Dirección del salto ("left" o "right")
        self.direction = direction

        # Nodo padre
        self.parent = parent

        # Hijos
        self.children = []

        # Puntuación de esta rama
        self.score = global_height

        # Indica si esta rama no debe seguir expandiéndose
        self.finished = False