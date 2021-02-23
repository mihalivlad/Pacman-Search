# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = successorGameState.getScore()
        score += currentGameState.hasFood(newPos[0], newPos[1]) * 100
        d1 = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        d2 = [manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()]
        distanceGhost = min(d2) if len(d2) > 0 else 0
        distanceFood = min(d1) if len(d1) > 0 else 0
        foodList = newFood.asList()
        currentPos = newPos
        a, b, c, d = currentPos, currentPos, currentPos, currentPos
        for nx, ny in foodList:
            if (nx <= a[0] and ny >= a[1]): a = (nx, ny)
            if (nx >= b[0] and ny >= b[1]): b = (nx, ny)
            if (nx <= c[0] and ny <= c[1]): c = (nx, ny)
            if (nx >= d[0] and ny <= d[1]): d = (nx, ny)

        distance = 0
        cornersList = [a, b, c, d]
        while len(cornersList) != 0:
            min2 = ((0, 0), 999999)
            for corner in cornersList:
                manD = (corner, util.manhattanDistance(corner, currentPos))
                if manD[1] < min2[1]:
                    min2 = manD
            distance += min2[1]
            currentPos = min2[0]
            cornersList.remove(currentPos)
        score -= distance
        score -= sum(d1) / 80
        score -= distanceFood * 2
        score += distanceGhost
        if distanceGhost < 2:
            score = -1e6

        if action == Directions.STOP:
            score -= 85
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return max(((action, self.MinValue(gameState.generateSuccessor(0, action)))
                    for action in gameState.getLegalActions(0)), key=lambda x: x[1])[0]

    def MaxValue(self, gameState, depth):
        if self.TerminalTest(depth, gameState):
            return self.evaluationFunction(gameState)

        return max([self.MinValue(gameState.generateSuccessor(0, action), depth)
                    for action in gameState.getLegalActions(0)])

    def MinValue(self, gameState, depth=0, agentIndex=1):
        if self.TerminalTest(depth, gameState, agentIndex):
            return self.evaluationFunction(gameState)
        if agentIndex == gameState.getNumAgents() - 1:
            return min([self.MaxValue(gameState.generateSuccessor(agentIndex, action), depth + 1)
                        for action in gameState.getLegalActions(agentIndex)])
        return min([self.MinValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                    for action in gameState.getLegalActions(agentIndex)])

    def TerminalTest(self, depth, gameState, agentIndex=0):
        return (depth == self.depth) or len(gameState.getLegalActions(agentIndex)) == 0


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        max_action = None
        for action in gameState.getLegalActions(0):
            action_value = self.MinValue(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if alpha < action_value:
                alpha = action_value
                max_action = action
        return max_action

    def MaxValue(self, gameState, depth, alpha, beta):
        if self.TerminalTest(depth, gameState):
            return self.evaluationFunction(gameState)
        v = float('-inf')
        for a in gameState.getLegalActions(0):
            v = max(v, self.MinValue(gameState.generateSuccessor(0, a), 1, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def MinValue(self, gameState, agentIndex, depth, alpha, beta):

        if self.TerminalTest(depth, gameState, agentIndex):
            return self.evaluationFunction(gameState)

        v = float('inf')
        for a in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                v = min(v, self.MaxValue(gameState.generateSuccessor(agentIndex, a), depth + 1, alpha, beta))
            else:
                v = min(v,
                        self.MinValue(gameState.generateSuccessor(agentIndex, a), agentIndex + 1, depth, alpha, beta))

            if v < alpha:
                return v
            beta = min(beta, v)

        return v

    def TerminalTest(self, depth, gameState, agentIndex=0):
        return (depth == self.depth) or len(gameState.getLegalActions(agentIndex)) == 0


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
