import copy
import random
import skeLCS.Classifier as Classifier
from skeLCS.TreePrint import build_tree_from_rpn


class CodeFragment:


    """
    GP tree node class:
    - value: node value (string), could be operator or variable name.
    - children: list of child nodes, unary operators (like sin) have 1 child, binary operators (like +, -, *, /) have 2 children.
    """
    def __init__(self,value, children=None, position = None):
        self.value = value
        self.children = children if children else []
        self.position = position

    def __str__(self):
        """
        Directly call toPostfix(), display postfix expression when printing
        """
        return self.toPostfix()

    def printTree(self):
        expr = self.toPostfix()
        tree = build_tree_from_rpn(expr)
        print(tree)

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

        # Theoretically shouldn't reach here
        return str(self.value)

    @staticmethod
    def createCodeFragment(variables, level=1):
        return CodeFragment._generateRandomTree(variables, max_level=level,current_level=level)

    @staticmethod
    def _generateRandomTree(variables,max_level,current_level=0,max_depth=2,current_depth=0):
        # Choose terminal: depth is 2(max), or random terminal probability 0.5
        if current_depth == Classifier.MAX_DEPTH or (random.random() > 0.5):
            # 1 level cf, just choose a feature
            if current_level == 1:
                position = random.choice(variables)
                return CodeFragment('D' + str(position), position=position)
            else:
                # max_level choose a feature and probability 0.5
                if max_level == current_level and random.random() > 0.5:
                    position = random.choice(variables)
                    return CodeFragment('D' + str(position), position=position)
                else:# otherwise,terminal choose a lower level cf.
                    lower_level = random.choice(list(range(1,current_level)))
                    child = CodeFragment._generateRandomTree(variables,max_level,current_level = lower_level)
                    return child

        # Randomly select an operator from OPERATOR_ARITY keys, then check arity
        op, arity = random.choice(list(Classifier.OPERATOR_ARITY.items()))

        if arity == 1:
            # Unary operator, e.g., sin
            child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level,current_depth= current_depth + 1)
            return CodeFragment(op, [child])
        elif arity == 2:
            # Binary operators +, -, *, /
            left_child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level, current_depth=current_depth + 1)
            right_child = CodeFragment._generateRandomTree(variables,max_level = max_level, current_level=current_level, current_depth=current_depth + 1)
            return CodeFragment(op, [left_child, right_child])
        else:
            raise Exception('Invalid arity')

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

