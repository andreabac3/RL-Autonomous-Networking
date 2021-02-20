from typing import List, Set

import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt


class AI_1747105(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)

        self.drone_zero = drone
        self.q_table: dict = {}  # cell -> drone -> q table value

        self.already_taken_actions = {}  # id_event (int) : {cell(int), set(droni assegnati))

        self.store_timestep_id_event = dict()  # id_event (int): {cell (int): timestep(int)}

        self.time_step: int = 0
        self.epsilon: float = 0.98

        # for metrics purpose
        self.count_exploration: int = 0
        self.count_exploitation: int = 0

    def _get_key_min_max_dictiory(self, dictionary: dict, func):
        return func(dictionary.items(), key=lambda x: x[1])[0]

    def _clean_already_taken_action(self, id_event, cell, drone):
        ''''we remove the old id_event for which we have already received all feedback'''
        self.already_taken_actions[id_event][cell].remove(drone)
        if len(self.already_taken_actions[id_event][cell]) == 0:
            del self.already_taken_actions[id_event][cell]
            if len(self.already_taken_actions[id_event].keys()) == 0:
                del self.already_taken_actions[id_event]

    def _get_right_cell_feedback(self, drone, id_event) -> int:

        '''
        This function is used to calculate the right cell for a drone in a particular id_event
        knowning that id_event can be sent in different cell
        '''
        list_state_explored = list(self.already_taken_actions[id_event].keys())
        if len(list_state_explored) == 1:
            '''
            (Base) Case in which the id_event is not sent in different cell
            '''
            unique_state = list_state_explored[0]
            return unique_state

        '''
        Case in which the id_event is sent over multiple cell
        We find the right cell for the drone given in input.
        '''

        seen = []
        for state in list_state_explored:
            if drone in self.already_taken_actions[id_event][state]:
                seen.append(state)
        if len(seen) == 1:
            '''
            Drone x is found in only one cell, so there's no inference
            '''
            unique_state = seen[0]
            return unique_state
        else:
            '''
            Case in which Drone x is reached in multiple cell for the same id_event
            we assign the drone in the first cell seen for giving priority.
            '''
            # there is overlap
            first_timestep = float("inf")
            best_state = None
            for state in seen:
                timestep = self.store_timestep_id_event[id_event][state]
                if timestep < first_timestep:
                    best_state = state
                    first_timestep = timestep
            return best_state

    def feedback(self, drone, id_event, delay, outcome):

        if drone == self.drone_zero:
            # we skip all feedback related to drone zero
            return None

        if id_event in self.already_taken_actions:

            cell: int = self._get_right_cell_feedback(drone=drone, id_event=id_event)  # take the right cell

            if drone not in self.q_table[cell]:
                # if the first time, initialize the q_table for the given cell
                self.q_table[cell][drone] = float('inf')

            # update the q-table
            self.q_table[cell][drone] = self.q_table[cell][drone] if delay > self.q_table[cell][drone] else delay

            self._clean_already_taken_action(id_event, cell, drone)  # we remove the old id_event for which we have already received all feedback

    def _store_map_event_to_cell(self, id_event: int, cell: int) -> None:
        '''
        This function is used to store the bijection between an id_event -> cell -> first time step seen
        '''
        if id_event not in self.store_timestep_id_event:
            # insert the first map between event -> cell -> timestep
            self.store_timestep_id_event[id_event] = {cell: self.time_step}
        elif cell not in self.store_timestep_id_event[id_event]:
            # we add another cell
            self.store_timestep_id_event[id_event][cell] = self.time_step

    def _store_action(self, id_event: int, cell: int, action) -> None:
        '''
        This function is used to store the decision made for an event in a particular cell for a drone
        We store this decision the a.t.a data structure
        '''
        if id_event not in self.already_taken_actions:
            self.already_taken_actions[id_event] = {cell: {action}}
        else:
            if cell in self.already_taken_actions[id_event]:
                self.already_taken_actions[id_event][cell].add(action)
            else:
                self.already_taken_actions[id_event][cell] = {action}

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        self.time_step += 1  # moving on time_step
        id_event: int = pkd.event_ref.identifier  # refactor for id_event
        id_set_neighbors: Set[int] = {v[1] for v in opt_neighbors}  # set of drone id

        cell: int = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell, width_area=self.simulator.env_width, x_pos=self.drone.coords[0], y_pos=self.drone.coords[1])[0])

        self._store_map_event_to_cell(id_event, cell)

        if cell not in self.q_table:
            # initialize q_table for the given cell if needed
            self.q_table[cell] = {}

        epsilon_choice: bool = self.rnd_for_routing_ai.random() > self.epsilon
        no_prior_knowledge_on_state: bool = len(self.q_table[cell].keys()) == 0  # The q table for the given cell is empty, so we make exploration
        we_do_exploration: bool = id_event in self.already_taken_actions and cell in self.already_taken_actions[id_event]

        if no_prior_knowledge_on_state or (we_do_exploration and epsilon_choice):
            self.count_exploration += 1
            # exploration
            set_id_drone_to_explore = id_set_neighbors  # first time drone to explore are equal to neighbors, then we remove the already explored
            if id_event in self.already_taken_actions:
                if cell in self.already_taken_actions[id_event]:
                    '''
                    For each action we store the biijection between the id_event and the selected actions (set of drones)
                    when we do exploration we don't want produce too many packets, so we take the set difference between
                    the actual neighbours (because drones are always in movement) and the already explored.
                    '''
                    already_explored: set = self.already_taken_actions[id_event][cell]  # we get the set of drones already explored in a given id_event
                    set_id_drone_to_explore = id_set_neighbors.difference(already_explored)  # difference between the set of actual neighbours - already drone explored
                    if len(set_id_drone_to_explore) == 0:
                        return None  # no action we have already explored all actual neighbours

            action = self.rnd_for_routing_ai.choice(list(set_id_drone_to_explore))

            self._store_action(id_event=id_event, action=action, cell=cell)

            return action
        else:
            # exploitation
            self.count_exploitation += 1

            sorted_q_table_for_cell = sorted(self.q_table[cell].items(), key=lambda kv: kv[1])  # for a given cell, we sort the drones by their q-table value
            best_action = None
            for best_drone, value in sorted_q_table_for_cell:
                is_drone_sucessfull: bool = value != self.simulator.event_duration
                if best_drone in id_set_neighbors and is_drone_sucessfull:
                    # we take the best drone w.r.t q-table and we discard the unsucessfull drones, which never get a success.
                    best_action = best_drone
                    break

            if best_action not in id_set_neighbors or best_action is None:
                return None  # no action

            self._store_action(id_event=id_event, action=best_action, cell=cell)

            return best_action  # return drone id

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        print("q table", self.q_table)
        print("EXPLOITATION", self.count_exploitation)
        print("EXPLORATION -> ", self.count_exploration)
