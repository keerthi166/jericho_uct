import sys
import math
import random


class State(object):#Done
    def equal(self, state):
        pass

    def duplicate(self):#Done
        pass

    def print(self):#Done
        pass

    def __del__(self):#Done
        pass


class SimAction(object):
    def equal(self, act):  # equal(SimAction* act) = 0;#Done
        pass

    def duplicate(self):#Done
        pass

    def printstate(self):#Done
        pass

    def __del__(self):#Done
        pass

class Simulator(object):
    def setState(self, state):#Done
        pass

    def getState(self, state):#Done
        pass

    def act(self, action):  # equal(SimAction* act) = 0;#Done

        pass

    def getActions(self):#Done
        pass

    def isTerminal(self):#Done
        pass

    def reset(self):#Done
        pass


class StateNode(object):

    def __init__(self, _parentAct, _state, _actVect, _reward, _isTerminal):#Done
        self.parentAct_ = _parentAct

        self.state_ = _state.duplicate()
        self.reward_ = _reward
        self.isTerminal_ = _isTerminal
        self.numVisits_ = 0
        self.actPtr_ = 0
        self.firstMC_ = 0
        self.actVect_ = []
        _size = len(_actVect)

        for i in range(0, _size):
            self.actVect_.append(_actVect[i].duplicate())
        random.shuffle(self.actVect_)
        #dictionary
        self.nodeVect_ = []  # vector<ActionNode*> nodeVect_;

    def __del__(self):#Done
        del self.state_
        #TODO dictionary change
        self.actVect_.clear()

    def isFull(self):#Done
        #TODO dictionary change
        return (self.actPtr_ == len(self.actVect_))

    def addActionNode(self):#Done
        assert (self.actPtr_ < len(self.actVect_))
        #TODO dict append
        self.nodeVect_.append(ActionNode(self))
        self.actPtr_ += 1
        return self.actPtr_ - 1


class ActionNode(object):#Done
    def __init__(self, _parentState):

        self.parentState_ = _parentState
        self.avgReturn_ = 0
        self.numVisits_ = 0
        self.stateVect_ = []

    def containNextState(self, state):
        #TODO class instances comparation,like find
        size = len(self.stateVect_)
        for i in range(0, size):
            if state.equal(self.stateVect_[i].state_):
                return True
        return False


    def getNextStateNode(self, state):
        #TODO class instances comparation,like find
        size = len(self.stateVect_)
        for i in range(0, size):
            #TODO dictionary search
            if state.equal(self.stateVect_[i].state_):
                return self.stateVect_[i]
        return None

    def addStateNode(self, _state, _actVect, _reward, _isTerminal):#Done
        index = len(self.stateVect_)
        #TODO dict append
        self.stateVect_.append(StateNode(self, _state, _actVect, _reward, _isTerminal))
        return self.stateVect_[index]

    def __del__(self):#Done
        pass


class UCTPlanner(object):

    def __init__(self, _sim, _maxDepth, _numRuns, _ucbScalar, _gamma, _leafValue, _endEpisodeValue):
        self.sim_ = _sim
        self.maxDepth_ = _maxDepth
        self.numRuns_ = _numRuns
        self.ucbScalar_ = _ucbScalar
        self.gamma_ = _gamma
        self.leafValue_ = _leafValue
        self.endEpisodeValue_ = _endEpisodeValue

        self.root_ = None

        _leafValue = 0
        _endEpisodeValue = 0

    def __del__(self):
        pass

    def setRootNode(self, _state, _actVect, _reward, _isTerminal):#Done
        if self.root_ != None:
            self.clearTree()
        self.root_ = StateNode(None, _state, _actVect, _reward, _isTerminal)

    def clearTree(self):
        self.root_ = None
        pass

    def plan(self):
        assert (self.root_ != None)
        #root offset is ????
        rootOffset = self.root_.numVisits_
        uctBranch = 0
        if rootOffset == 0:
            self.root_.numVisits_ += 1
            rootOffset += 1
        for trajectory in range(rootOffset, self.numRuns_):
            current = self.root_
            mcReturn = self.leafValue_
            depth = 0
            while True:
                depth += 1
                #is current is terminal, we need to stop
                if current.isTerminal_:
                    mcReturn = self.endEpisodeValue_
                    break
                #if is full,means we have already tried every action,
                #we need to find the 'best' child
                if current.isFull():
                    uctBranch = 0
                    if current == self.root_:
                        uctBranch = self.getUCTRootIndex(current)
                    else:
                        uctBranch = self.getUCTBranchIndex(current)
                    self.sim_.setState(current.state_)
                    #after select the highest UCB action we need to act it, and find the state we need
                    #TODO dict
                    r = self.sim_.act(current.actVect_[uctBranch])

                    nextState = self.sim_.getState()
                    #we need this  if-else below for the general case:
                    # It is possible that, although this action node has been expended before,
                    # but the distribution of transition is not deterministic, so possible that
                    # the state node that we act on s has not been generated before
                    # so it is not in the state list of the current action node
                    if current.nodeVect_[uctBranch].containNextState(nextState):
                        # follow path
                        current = current.nodeVect_[uctBranch].getNextStateNode(nextState)
                        continue
                    else:
                        #TODO dict
                        nextNode = current.nodeVect_[uctBranch].addStateNode(nextState, self.sim_.getActions(), r,
                                                                             self.sim_.isTerminal())
                        if -1 == self.maxDepth_:
                            mcReturn = self.MC_Sampling_terminal(nextNode)
                        else:
                            mcReturn = self.MC_Sampling_depth(nextNode, self.maxDepth_ - depth)
                        current = nextNode
                        break
                else:
                    #ptr used to store the the current length
                    actID = current.addActionNode()
                    #the current now is a state node,so all the childen are action node
                    #randomly choose an an action and return the reward
                    self.sim_.setState(current.state_)
                    r = self.sim_.act(current.actVect_[actID])
                    #current.nodeVect_are actions, not states
                    #the state in the simulator has changed, but current has not
                    nextNode = current.nodeVect_[actID].addStateNode(self.sim_.getState(), self.sim_.getActions(), r,self.sim_.isTerminal())
                    if -1 == self.maxDepth_:
                        mcReturn = self.MC_Sampling_terminal(nextNode)
                    else:
                        mcReturn = self.MC_Sampling_depth(nextNode, self.maxDepth_ - depth)
                    current = nextNode
                    break

            self.updateValues(current, mcReturn)

    def getAction(self):
        return self.root_.actVect_[self.getGreedyBranchIndex()]

    def getMostVisitedBranchIndex(self):
        assert (self.root_ != None)

        maximizer = []
        size = len(self.root_.nodeVect_)
        #TODO dict max
        for i in range(0, size):
            maximizer.append(self.root_.nodeVect_[i].numVisits_)
        return maximizer.index(max(maximizer))

    def getGreedyBranchIndex(self):
        assert (self.root_ is not None)
        maximizer = []
        size = len(self.root_.nodeVect_)
        for i in range(0, size):
            maximizer.append(self.root_.nodeVect_[i].avgReturn_)
        return maximizer.index(max(maximizer))

    def getUCTRootIndex(self, node):#Done
        det = math.log(float(node.numVisits_))
        #maximizer: the list of the max UCB
        maximizer = []  # maximizer.clear()
        size = len(node.nodeVect_)
        for i in range(0, size):
            val = node.nodeVect_[i].avgReturn_
            val += self.ucbScalar_ * math.sqrt(det / float(node.nodeVect_[i].numVisits_))
            maximizer.append(val)
        return maximizer.index(max(maximizer))

    # same as before
    def getUCTBranchIndex(self, node):
        det = math.log(float(node.numVisits_))
        maximizer = []
        size = len(node.nodeVect_)
        for i in range(0, size):
            val = node.nodeVect_[i].avgReturn_
            val += self.ucbScalar_ * math.sqrt(det / float(node.nodeVect_[i].numVisits_))

            maximizer.append(val)

        return maximizer.index(max(maximizer))

    def updateValues(self, node, mcReturn):
        totalReturn = mcReturn
        if node.numVisits_ == 0:
            node.firstMC_ = totalReturn

        node.numVisits_ += 1
         # back until root is reached, the parent of root is None
        while node.parentAct_ is not None:
            parentAct = node.parentAct_
            parentAct.numVisits_ += 1
            totalReturn *= self.gamma_
            totalReturn += self.modifyReward(node.reward_)
            # avg = (total+avg0(n-1))/n
            # avg = avg0+(total-avg0)/n
            parentAct.avgReturn_ += (totalReturn - parentAct.avgReturn_) / parentAct.numVisits_
            node = parentAct.parentState_
            node.numVisits_ += 1
    def MC_Sampling_depth(self, node, depth):
        mcReturn = self.leafValue_
        self.sim_.setState(node.state_)
        discnt = 1
        for i in range(0, depth):
            if self.sim_.isTerminal():
                mcReturn += self.endEpisodeValue_
                break
            #TODO random in dict
            actions = self.sim_.getActions()
            actID = int(random.random()*len(actions))
            r = self.sim_.act(actions[actID])
            mcReturn += discnt * self.modifyReward(r)
            discnt *= self.gamma_
        return mcReturn

    def MC_Sampling_terminal(self, node):
        mcReturn = self.endEpisodeValue_
        self.sim_.setState(node.state_)
        discnt = 1
        while not self.sim_.isTerminal():
            actions = self.sim_.getActions()
            actID = int(random.random()*len(actions))
            r = self.sim_.act(actions[actID])
            mcReturn += discnt * self.modifyReward(r)
            discnt *= self.gamma_
        return mcReturn

    def modifyReward(self, orig):
        return orig

    def printRootValues(self):
        size = len(self.root_.nodeVect_)
        for i in range(0, size):
            val = self.root_.nodeVect_[i].avgReturn_
            numVist = self.root_.nodeVect_[i].avgReturn_
            print("(", self.root_.actVect_.printact(), ",", val, ",", numVist, ") ")
        print(self.root_.isTerminal_)

    def clearTree(self):
        if self.root_ is not None:
            self.pruneState(self.root_)
        self.root_ = None

    def terminalRoot(self):
        return self.root_.isTerminal_

    #TODO prune
    def prune(self, act):
        #TODO dict
        nextRoot = None
        size = len(self.root_.nodeVect_)
        for i in range(0, size):
            if act.equal(self.root_.actVect_[i]):
                assert (len(self.root_.nodeVect_[i].stateVect_) == 1)
                nextRoot = self.root_.nodeVect_[i].stateVect_[0]
                tmp = self.root_.nodeVect_[i]
                del tmp
            else:
                tmp = self.root_.nodeVect_[i]
                self.pruneAction(tmp)

        assert (nextRoot != None)
        self.root_ = nextRoot
        self.root_.parentAct_ = None

    def pruneState(self, state):
        #TODO dict
        sizeNode = len(state.nodeVect_)
        for i in range(0, sizeNode):
            tmp = state.nodeVect_[i]
            self.pruneAction(tmp)

        state.nodeVect_ = []
        del state


    def pruneAction(self, act):
        #TODO dict
        sizeNode = len(act.stateVect_)
        for i in range(0, sizeNode):
            tmp = act.stateVect_[i]
            self.pruneState(tmp)
        act.stateVect_ = []
        del act

    def testRoot(self, _state, _reward, _isTerminal):
        return self.root_ != None \
               and (self.root_.reward_ == _reward) \
               and (self.root_.isTerminal_ == _isTerminal) \
               and self.root_.state_.equal(_state)

    def testDeterministicProperty(self):
        if self.testDeterministicPropertyState(self.root_):
            print("Deterministic Property Test passed!")
        else:
            print("Error in Deterministic Property  Test!")
            sys.exit(0)

    def testDeterministicPropertyState(self, state):
        actSize = len(state.nodeVect_)
        # we test all the actions under a state
        for i in range(0, actSize):
            if not self.testTreeStructureAction(state.nodeVect_[i]):
                return False
        return True

    def testDeterministicPropertyAction(self, action):
        stateSize = len(action.stateVect_)
        #under a deterministic proerty, a on s can only genreate one sperrcific s'
        if stateSize != 1:
            print("Error in Deterministic Property Test!")
            return False
        # test every state under an action
        for i in range(0, stateSize):
            #TODO dict
            if not self.testTreeStructureState(action.stateVect_[i]):
                #print ("actiontest:Flase")
                return False;
        #print("actiontest:True")
        return True

    def testTreeStructure(self):
        if self.testTreeStructureState(self.root_):
            print("Tree Structure Test passed!")
        else:
            print("Error in Tree Structure Test!")
            sys.exit(1)

    def testTreeStructureState(self, state):
        actVisitCounter = 0
        actSize = len(state.nodeVect_)
        #TODO dict
        for i in range(0, actSize):
            actVisitCounter += state.nodeVect_[i].numVisits_;
        #find out that whether the total number of states' visit is
        # equal to the number of the visit of their parent action
        if (actVisitCounter + 1 != state.numVisits_) and (not state.isTerminal_):
            print("n(s) = sum_{a} n(s,a) + 1 failed ! \n Diff: " \
                  , actVisitCounter + 1 - state.numVisits_, \
                  "\nact: ", actVisitCounter + 1, "\nState: ", \
                  state.numVisits_, "\nTerm: ", \
                  state.isTerminal_, "\nState: ")
            state.state_.print()
            print("")
            return False

        for i in range(0, actSize):
            #TODO dict
            if not self.testTreeStructureAction(state.nodeVect_[i]):
                return False

        return True

    def testTreeStructureAction(self, action):
        stateVisitCounter = 0
        stateSize = len(action.stateVect_)
        #TODO dict
        for i in range(0, stateSize):
            stateVisitCounter += action.stateVect_[i].numVisits_

        if stateVisitCounter != action.numVisits_:
            print("n(s,a) = sum n(s') failed !")
            return False
        # avg
        # Q(s,a) = E {r(s') + gamma * sum pi(a') Q(s',a')}
        # Q(s,a) = sum_{s'} n(s') / n(s,a) * ( r(s') + gamma * sum_{a'} (n (s',a') * Q(s',a') + first) / n(s'))
        value = 0
        for i in range(0, stateSize):
            next = action.stateVect_[i]
            w = next.numVisits_ / float(action.numVisits_)
            nextValue = next.firstMC_
            nextActSize = len(next.nodeVect_)
            for j in range(0, nextActSize):
                nextValue += next.nodeVect_[j].numVisits_ * next.nodeVect_[j].avgReturn_
            nextValue = (nextValue) / next.numVisits_ * self.gamma_
            nextValue += next.reward_
            value += w * nextValue

        if (action.avgReturn_ - value) * (action.avgReturn_ - value) > 1e-10:
            print("value constraint failed !", \
                  "avgReturn=", action.avgReturn_, " value=", value)
            return False

        for i in range(0, stateSize):
            if not self.testTreeStructureState(action.stateVect_[i]):
                return False
        #print("testTreeStructureAction pass")
        return True
