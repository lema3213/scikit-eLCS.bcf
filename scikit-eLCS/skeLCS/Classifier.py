import random
import copy
import math

from skeLCS.CodeFragment import CodeFragment
from skeLCS.Condition import Condition

OPERATOR_ARITY = {
    '&': 2,
    '|': 2,
    '~': 1,
    'nand': 2,
    'nor': 2,
}

MAX_DEPTH = 2

class Classifier:

    def __init__(self,elcs,a=None,b=None,c=None,d=None):
        #Major Parameters
        self.condition = []
        self.phenotype = None #arbitrary

        self.fitness = elcs.init_fit
        self.accuracy = 0.0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        # Experience Management
        self.timeStampGA = None
        self.initTimeStamp = None

        # Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0  # Known in many LCS implementations as experience i.e. the total number of times this classifier was in a match set
        self.correctCount = 0  # The total number of times this classifier was in a correct set

        if isinstance(c, list):
            self.classifierCovering(elcs, a, b, c, d)
        elif isinstance(a, Classifier):
            self.classifierCopy(a, b)

    # Classifier Construction Methods
    def classifierCovering(self, elcs, setSize, exploreIter, state, phenotype):
        # Initialize new classifier parameters----------
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = setSize
        dataInfo = elcs.env.formatData

        # -------------------------------------------------------
        # DISCRETE PHENOTYPE
        # -------------------------------------------------------
        if dataInfo.discretePhenotype:
            self.phenotype = phenotype
        # -------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        # -------------------------------------------------------
        else:
            phenotypeRange = dataInfo.phenotypeList[1] - dataInfo.phenotypeList[0]
            rangeRadius = random.randint(25,75) * 0.01 * phenotypeRange / 2.0  # Continuous initialization domain radius.
            Low = float(phenotype) - rangeRadius
            High = float(phenotype) + rangeRadius
            self.phenotype = [Low, High]

        while len(self.getWorkingCondition()) < 1:
            for attRef in range(len(state)):
                condition = None
                if random.random() < elcs.p_spec:
                    condition = self.buildMatch(elcs,state)  # Add classifierConditionElement
                if condition is None:
                    condition = Condition()
                self.condition.append(condition)
    def classifierCopy(self, toCopy, exploreIter):
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = exploreIter
        self.initTimeStamp = exploreIter
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def buildMatch(self, elcs, state):
        attributes = list(range(0, elcs.env.formatData.numAttributes))
        for i in range(100):

            cf = CodeFragment.createCodeFragment(variables=attributes, level=elcs.level)

            result = CodeFragment.evaluate( cf, state)

            if result ==1:
                condition = Condition(cf)
            else:
                continue
            # Check if current rule already contains this expression
            if condition.expression in [cd.expression for cd in self.condition if not cd.is_dc]:
                continue

            return condition
        return None

    # Matching
    def match(self, state, elcs):
        condition = self.condition

        for cd in condition:
            if cd.is_dc:
                continue
            result = CodeFragment.evaluate(cd.codeFragment, state)

            if result != 1:
                return False

        return True

    def equals(self, elcs, cl):
        if cl.phenotype == self.phenotype and len(cl.getWorkingCondition()) == len(self.getWorkingCondition()):
            for cd in cl.condition:
                if cd.is_dc:
                    continue

                idx = self.findIndexByExpression(self, cd.expression)
                if idx is None or str(cd) != str(self.condition[idx]):
                    return False
            return True

        return False

    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num

    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1

    def updateCorrect(self):
        """ Increases the correct phenotype tracking by one. Once an epoch has completed, rule accuracy can't change."""
        self.correctCount += 1

    def updateMatchSetSize(self, elcs, matchSetSize):
        """  Updates the average match set size. """
        if self.matchCount < 1.0 / elcs.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount - 1) + matchSetSize) / float(
                self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + elcs.beta * (matchSetSize - self.aveMatchSetSize)

    def updateAccuracy(self):
        """ Update the accuracy tracker """
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self, elcs):
        """ Update the fitness parameter. """
        if elcs.env.formatData.discretePhenotype or (
                self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange < 0.5:
            self.fitness = pow(self.accuracy, elcs.nu)
        else:
            if (self.phenotype[1] - self.phenotype[0]) >= elcs.env.formatData.phenotypeRange:
                self.fitness = 0.0
            else:
                self.fitness = math.fabs(pow(self.accuracy, elcs.nu) - (
                            self.phenotype[1] - self.phenotype[0]) / elcs.env.formatData.phenotypeRange)

    def isSubsumer(self, elcs):
        if self.matchCount > elcs.theta_sub and self.accuracy > elcs.acc_sub:
            return True
        return False

    def isMoreGeneral(self, cl, elcs):
        if len(self.getWorkingCondition()) >= len(cl.getWorkingCondition()):
            return False

        for i in range(len(self.condition)):
            cd = self.condition[i]
            if cd.is_dc:
                continue
            clIndex = self.findIndexByExpression(cl, cd.expression)
            if clIndex is None:
                return False

        return True

    def uniformCrossover(self, elcs, cl):
        changed = False
        x = random.randint(0, len(self.condition))
        y = random.randint(0, len(cl.condition))

        if x > y:
            x, y = y, x

        for i in range(x, y):
            if str(self.condition[i]) != str(cl.condition[i]):
                self.condition[i], cl.condition[i] = cl.condition[i], self.condition[i]
                changed = True
        return changed

    def phenotypeCrossover(self, cl):
        changed = False
        if self.phenotype == cl.phenotype:
            return changed
        else:
            tempKey = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey:  # Swap minimum
                temp = self.phenotype[0]
                self.phenotype[0] = cl.phenotype[0]
                cl.phenotype[0] = temp
                changed = True
            elif tempKey:  # Swap maximum
                temp = self.phenotype[1]
                self.phenotype[1] = cl.phenotype[1]
                cl.phenotype[1] = temp
                changed = True

        return changed

    def Mutation(self, elcs, state, phenotype):
        changed = False
        # Mutate Condition
        for i in range(len(self.condition)):
            cd = self.condition[i]
            if random.random() < elcs.mu:
                # Mutation
                if cd.is_dc:
                    # dc -> cf
                    condition = self.buildMatch(elcs, state)
                    if condition is not None:
                        self.condition[i] = condition
                        changed = True
                else:
                    # cf -> dc
                    if random.random() > 0.5:
                        self.condition[i]= Condition()
                        changed = True
                    else: # cf -> new cf
                        condition = self.buildMatch(elcs, state)
                        if condition is not None:
                            self.condition[i] = condition
                            changed = True

        # Mutate Phenotype
        if elcs.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation(elcs)

        if changed or nowChanged:
            return True


    def discretePhenotypeMutation(self, elcs):
        changed = False
        if random.random() < elcs.mu:
            phenotypeList = copy.deepcopy(elcs.env.formatData.phenotypeList)
            phenotypeList.remove(self.phenotype)
            newPhenotype = random.choice(phenotypeList)
            self.phenotype = newPhenotype
            changed = True
        return changed

    def continuousPhenotypeMutation(self, elcs, phenotype):
        changed = False
        if random.random() < elcs.mu:
            phenRange = self.phenotype[1] - self.phenotype[0]
            mutateRange = random.random() * 0.5 * phenRange
            tempKey = random.randint(0,2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0:  # Mutate minimum
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                changed = True
            elif tempKey == 1:  # Mutate maximum
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.phenotype[0] + mutateRange <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[0] += mutateRange
                else:  # Subtract
                    self.phenotype[0] -= mutateRange
                if random.random() > 0.5 or self.phenotype[1] - mutateRange >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.phenotype[1] -= mutateRange
                else:  # Subtract
                    self.phenotype[1] += mutateRange
                changed = True
            self.phenotype.sort()
        return changed

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def subsumes(self, elcs, cl):
        # Discrete Phenotype
        if elcs.env.formatData.discretePhenotype:
            if cl.phenotype == self.phenotype:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
            return False

        # Continuous Phenotype
        else:
            if self.phenotype[0] >= cl.phenotype[0] and self.phenotype[1] <= cl.phenotype[1]:
                if self.isSubsumer(elcs) and self.isMoreGeneral(cl, elcs):
                    return True
                return False

    def getDelProp(self, elcs, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= elcs.delta * meanFitness or self.matchCount < elcs.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (elcs.init_fit / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote

    def findIndexByExpression(self, cl, expression):
        for i, cond in enumerate(cl.condition):
            if cond.expression == expression:
                return i
        return None

    def getWorkingCondition(self):
        return [cd for cd in self.condition if not cd.is_dc]