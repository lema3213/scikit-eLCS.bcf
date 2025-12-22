import random
import re

import pandas as pd

import skeLCS.Classifier as Classifier
from skeLCS.TreePrint import build_tree_from_rpn
class CodeFragment:
    OPERATOR_ARITY = {
        '&': 2,
        '|': 2,
        '~': 1,
        'nand': 2,
        'nor': 2,
    }

    MAX_DEPTH = 2

    path_l1 = r"../MetaData/CF_L1.csv"
    path_l2 = r"../MetaData/CF_L2.csv"
    path_l3 = r"../MetaData/CF_L3.csv"
    path_l4 = r"../MetaData/CF_L4.csv"
    path_l5 = r"../MetaData/CF_L5.csv"
    path_l6 = r"../MetaData/CF_L6.csv"

    try:
        df = pd.read_csv(path_l1)
        CF_L1 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L1 = []

    try:
        df = pd.read_csv(path_l2)
        CF_L2 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L2 = []

    try:
        df = pd.read_csv(path_l3)
        CF_L3 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L3 = []

    try:
        df = pd.read_csv(path_l4)
        CF_L4 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L4 = []

    try:
        df = pd.read_csv(path_l5)
        CF_L5 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L5 = []

    try:
        df = pd.read_csv(path_l6)
        CF_L6 = df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        CF_L6 = []



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
        if current_depth == CodeFragment.MAX_DEPTH or (random.random() > 0.5):
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

                    if lower_level == 1 and CodeFragment.CF_L1:
                        postfix = random.choice(CodeFragment.CF_L1)
                        child = CodeFragment.fromPostfix(postfix)
                    elif lower_level == 2 and CodeFragment.CF_L2:
                        postfix = random.choice(CodeFragment.CF_L2)
                        child = CodeFragment.fromPostfix(postfix)
                    elif lower_level == 3 and CodeFragment.CF_L3:
                        postfix = random.choice(CodeFragment.CF_L3)
                        child = CodeFragment.fromPostfix(postfix)
                    elif lower_level == 4 and CodeFragment.CF_L4:
                        postfix = random.choice(CodeFragment.CF_L4)
                        child = CodeFragment.fromPostfix(postfix)
                    elif lower_level == 5 and CodeFragment.CF_L5:
                        postfix = random.choice(CodeFragment.CF_L5)
                        child = CodeFragment.fromPostfix(postfix)
                    elif lower_level == 6 and CodeFragment.CF_L6:
                        postfix = random.choice(CodeFragment.CF_L6)
                        child = CodeFragment.fromPostfix(postfix)
                    else:
                        child = CodeFragment._generateRandomTree(variables,max_level,current_level = lower_level)
                    return child

        # Randomly select an operator from OPERATOR_ARITY keys, then check arity
        op, arity = random.choice(list(CodeFragment.OPERATOR_ARITY.items()))

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
        arity = CodeFragment.OPERATOR_ARITY[op]

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
                return 0 if (left_val == 1 and right_val == 1) else 1
            elif op == 'nor':
                return 1 if (left_val == 0 and right_val == 0) else 0

        elif arity == 4 and op == 'f':
            forth_val = CodeFragment.evaluateTree(cf.children[3], state)
            third_val = CodeFragment.evaluateTree(cf.children[2], state)
            second_val = CodeFragment.evaluateTree(cf.children[1], state)
            first_val = CodeFragment.evaluateTree(cf.children[0], state)
            result = third_val if first_val > second_val else forth_val
            return result

    @staticmethod
    def fromPostfix(postfix: str):
        """
        Parse a postfix (RPN) string back into a CodeFragment tree.

        Terminals:
          - D<int> (e.g., D3, D16) -> leaf node, position=<int>

        Operators:
          - Must exist in CodeFragment.OPERATOR_ARITY
          - Unary operators pop 1 operand
          - Binary operators pop 2 operands (order matters: left then right)

        Returns:
          - The root CodeFragment node

        Raises:
          - ValueError for invalid postfix strings or unknown tokens
        """
        if postfix is None:
            raise ValueError("postfix is None")

        tokens = postfix.strip().split()
        if not tokens:
            raise ValueError("Empty postfix string")

        term_pat = re.compile(r"^D(\d+)$")
        stack = []

        for tok in tokens:
            # Operator token
            if tok in CodeFragment.OPERATOR_ARITY:
                arity = CodeFragment.OPERATOR_ARITY[tok]
                if len(stack) < arity:
                    raise ValueError(f"Invalid postfix: not enough operands for operator '{tok}'")

                if arity == 1:
                    child = stack.pop()
                    stack.append(CodeFragment(tok, [child]))

                elif arity == 2:
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(CodeFragment(tok, [left, right]))

                else:
                    # Reserved for future multi-arity operators (e.g., arity=4)
                    children = [stack.pop() for _ in range(arity)][::-1]
                    stack.append(CodeFragment(tok, children))

                continue

            # Terminal token (leaf), e.g., D16
            m = term_pat.match(tok)
            if m:
                pos = int(m.group(1))
                stack.append(CodeFragment('D' + str(pos), position=pos))
                continue

            raise ValueError(f"Unknown token in postfix: '{tok}'")

        if len(stack) != 1:
            raise ValueError(f"Invalid postfix: stack has {len(stack)} items after parsing (expected 1)")

        return stack[0]