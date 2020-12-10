import uct
from copy import deepcopy
import hashlib
from jericho import *
from env.utils import make_env


# def _get_world_state_hash(env):
#     world_str = ', '.join([str(o) for o in env.get_world_objects()])
#     m = hashlib.md5()
#     m.update(world_str.encode('utf-8'))
#     return m.hexdigest()


class JerichoState(object):
    def __init__(self, state, done, actions):
        self.state = deepcopy(state)
        self.done = done
        self.actions = actions

    def equal(self, other):
        # print('state', self.state)
        # print('other', other.state)
        # print('1', all(self.state[0] == other.state[0]))
        return (all(self.state[0] == other.state[0]) and
                all(self.state[1] == other.state[1]) and
                self.state[2] == other.state[2] and
                self.state[3] == other.state[3] and
                self.state[4] == other.state[4] and
                self.state[5] == other.state[5] and
                self.done == other.done)

    def duplicate(self):#Done
        return JerichoState(self.state, self.done, self.actions)

    def print(self):
        return

    def __del__(self):#Done
        return


class JerichoAction(object):
    def __init__(self, action_text):
        self.action_text = action_text

    def equal(self, other):  # equal(SimAction* act) = 0;#Done
        return self.action_text == other.action_text

    def duplicate(self):#Done
        return JerichoAction(self.action_text)

    def __del__(self):#Done
        pass


class JerichoSimulator(object):
    def __init__(self,
                 rom_path="../env/roms/autoplay-game-suite/zork1.z5"):
        rom_path = "../env/roms/autoplay-game-suite/zork1.z5"
        bindings = load_bindings(rom_path)
        seed = bindings['seed']
        env = make_env(rom_path, seed, 100)
        self.env = env
        # max_word_len = bindings['max_word_length']
        # vocab = env.get_id2act_word()
        # vocab_rev = env.get_act_word2id()
        self.current_state = None
        self.reset()
        # JerichoState(self.env.get_state(), False)

    def setState(self, state):
        self.current_state = state
        self.env.set_state(self.current_state.state)
        pass

    def getState(self):#Done
        return self.current_state

    def act(self, action, return_text=False):  # equal(SimAction* act) = 0;#Done
        next_obs_text, reward, done, next_info = self.env.step(
            action.action_text, parallel=True)
        self.current_state = JerichoState(
            self.env.get_state(), done, next_info['valid_act'])
        if not return_text:
            return reward
        return reward, next_obs_text

    def getActions(self):#Done
        # tricky, construct to action list
        actions = self.current_state.actions
        actions = [act[0].action for act in actions]
        # print(actions)
        acts = [JerichoAction(at) for at in actions]
        # print(acts)
        return acts

    def isTerminal(self):#Done
        return self.current_state.done

    def reset(self, return_text=False):#Done
        obs, info = self.env.reset(parallel=True)
        self.current_state = JerichoState(
            self.env.get_state(), False, info['valid_act'])
        if not return_text:
            return
        return obs


if __name__ == '__main__':
    max_depth = 500
    num_runs = 800
    jericho_simulator = JerichoSimulator()
    jericho_simulator_uct = JerichoSimulator()
    uctTree = uct.UCTPlanner(jericho_simulator_uct, max_depth, num_runs,
                             0.95, 0.5, 0, 0)
    r = 0
    # sim.getState().print()
    step = 0
    total_r = 0
    text = jericho_simulator.reset(return_text=True)

    while not jericho_simulator.isTerminal():
        uctTree.setRootNode(jericho_simulator.getState(),
                            jericho_simulator.getActions(), r,
                            jericho_simulator.isTerminal())
        uctTree.plan()
        action = uctTree.getAction()

        r, next_text = jericho_simulator.act(action, return_text=True)

        total_r += r
        step += 1
        print("=" * 77)
        print("({}, obs) {}".format(step, text))
        print("({}) r={}, R={}, act={}".format(step, r, total_r,
                                               action.action_text))
        text = next_text

    print('total rewards:', total_r, "steps:", step)