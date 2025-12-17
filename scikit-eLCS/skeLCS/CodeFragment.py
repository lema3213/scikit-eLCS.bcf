import copy
import random
import skeLCS.Classifier as Classifier


class CodeFragment:


    """
    GP tree node class:
    - value: node value (string), could be operator or variable name.
    - children: list of child nodes, unary operators (like sin) have 1 child, binary operators (like +, -, *, /) have 2 children.
    """
    def __init__(self,value, children=None, level=0,position = None):
        self.value = value
        self.children = children if children else []
        self.level = level
        self.position = position

    def __str__(self):
        """
        Directly call toPostfix(), display postfix expression when printing
        """
        return self.toPostfix()

    def toPostfix(self):
        """
        Return postfix expression string (RPN).
        Using post-order traversal: traverse child nodes first, then add current node's operator.
        """
        if not self.children:
            # No children, means it's a variable
            return str(self.value)

        # If it's a unary operator
        if len(self.children) == 1:
            return f"{self.children[0].toPostfix()} {self.value}"

        # If it's a binary operator
        if len(self.children) == 2:
            left_expr = self.children[0].toPostfix()
            right_expr = self.children[1].toPostfix()
            return f"{left_expr} {right_expr} {self.value}"

        # If it's a binary operator
        if len(self.children) == 4:
            first_expr = self.children[0].toPostfix()
            second_expr = self.children[1].toPostfix()
            third_expr = self.children[2].toPostfix()
            forth_expr = self.children[3].toPostfix()
            return f"{first_expr} {second_expr} {third_expr} {forth_expr} {self.value}"

        # Theoretically shouldn't reach here
        return str(self.value)

    @staticmethod
    def createCodeFragment(variables, max_depth):
        return CodeFragment._generateRandomTree(variables, max_depth)

    @staticmethod
    def _generateRandomTree(variables, max_depth=0, current_depth=0):
        if current_depth == 0:
            variables = copy.deepcopy(variables)

        if current_depth == max_depth or random.random() > 0.5:
            position = random.choice(variables)
            return CodeFragment('D'+str(position),level=current_depth,position=position)

        # Randomly select an operator from OPERATOR_ARITY keys, then check arity
        op, arity = random.choice(list(Classifier.OPERATOR_ARITY.items()))

        if arity == 1:
            # Unary operator, e.g., sin
            child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            return CodeFragment(op, [child],level=current_depth)
        elif arity == 2:
            # Binary operators +, -, *, /
            left_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            right_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            return CodeFragment(op, [left_child, right_child], level=current_depth)
        elif arity == 4:
            first_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            second_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            third_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            forth_child = CodeFragment._generateRandomTree(variables, max_depth, current_depth + 1)
            return CodeFragment(op, [first_child, second_child, third_child, forth_child], level=current_depth)

    @staticmethod
    def evaluate(cf, state):
        return CodeFragment.evaluateTree(cf, state)

    @staticmethod
    def evaluateTree(cf, state):
        # If it's a leaf node (variable), directly get value from context
        if not cf.children:
            return state[cf.position]

        # Otherwise it's an operator node
        op = cf.value
        arity = Classifier.OPERATOR_ARITY[op]

        if arity == 1:
            # Unary operator (sin)
            val = CodeFragment.evaluateTree(cf.children[0], state)
            if op == '~':
                return 1 if (val == 0) else 0

        elif arity == 2:
            # Binary operators (+, -, *, /)
            left_val = CodeFragment.evaluateTree(cf.children[0], state)
            right_val = CodeFragment.evaluateTree(cf.children[1], state)

            if op == '&':
                return 1 if (left_val == 1 and right_val == 1) else 0
            elif op == '|':
                return 1 if (left_val == 1 or right_val == 1) else 0
            elif op == 'nand':
                return 1 if (left_val == 1 and right_val == 1) else 1
            elif op == 'nor':
                return 1 if (left_val == 0 and right_val == 0) else 0

        elif arity == 4 and op == 'f':
            forth_val = CodeFragment.evaluateTree(cf.children[3], state)
            third_val = CodeFragment.evaluateTree(cf.children[2], state)
            second_val = CodeFragment.evaluateTree(cf.children[1], state)
            first_val = CodeFragment.evaluateTree(cf.children[0], state)
            result = third_val if first_val > second_val else forth_val
            return result

