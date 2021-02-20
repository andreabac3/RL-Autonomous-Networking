import itertools

from src.entities.uav_entities import Drone
from src.mac_protocol.depot_mac import DepotMAC
import numpy as np

"""
The class is responsable to allocate communication resources to neighbors drones that want to offload data to the depot.
We work over an semplified TDMA approach, each time step only one drone can receive the resource and communicate a packet to the depot. 
"""


class OverlapMAC(DepotMAC):

    def __init__(self, simulator, depot):
        super().__init__(simulator, depot)
        self.rnd_mac = np.random.RandomState(self.simulator.seed)

        self.list_drone_feedback = {}  # dictionary that save the feedback received for each drone
        self.counter_frequency = {}  # dictionary that count the number of True transmission for each drone
        self.schedule = []  # learned pattern
        self.always_best: list = None  # set of best drones
        self.schedule_counter = 0
        self.epsilon = 0.2  # epsilon variable

        # learning variables
        self.number_of_drones = self.simulator.n_drones
        self.learn_actual_drone_id: int = 0
        self.duration_simulation: int = self.simulator.len_simulation  # simulation duration
        self.learning_duration = 0.1 * self.duration_simulation  # learning phase duration
        self.drone_step = int(
            self.learning_duration // self.number_of_drones)  # number of learning step dedicated for each drone

        self.module = self.drone_step - 1

    def _count_frequency(self):
        # function that count for each drone the true transmission from
        # list_drone_feedback and save it in counter_frequency dictionary
        for drone_sequence in self.list_drone_feedback.keys():
            count = 0
            for step in self.list_drone_feedback[drone_sequence]:
                if step != -1:
                    count += 1
            self.counter_frequency[drone_sequence] = count

    def _take_best(self) -> list:
        # function that take the best drones w.r.t. higher frequency
        best: list = []
        s = sum(self.counter_frequency.values())
        counter_frequency_per = [((cf / s) * 100) for cf in self.counter_frequency.values()]
        _max = max(counter_frequency_per)
        drone_best_id = counter_frequency_per.index(_max)
        for i, cfp in enumerate(counter_frequency_per):
            if abs(cfp - counter_frequency_per[drone_best_id]) <= 0.35 * _max:
                best.append(i)

        return best

    def allocate_resource_to_drone(self, drones: list, cur_step: int) -> Drone:
        """ Return the drone to who allocate bandwith for upload data in this step """

        if self.last_feedback != None:
            (drone, transmission, feedback) = self.last_feedback

        # initialization step
        if cur_step < 1:
            for drone in drones:
                self.list_drone_feedback[drone.identifier] = []
            return drones[0]

        # learning phase
        if cur_step < self.learning_duration:
            actual_drone = int(cur_step // self.drone_step)
            self.list_drone_feedback[drones[actual_drone].identifier].append(actual_drone if transmission else -1)
            return drones[actual_drone]

        else:
            # end of the learning phase, we use this "if" only one time after learning
            if cur_step == self.learning_duration:
                self._count_frequency()
                self.always_best: list = self._take_best()

                # we produce a schedule based on the learned drones behavior
                for i in range(self.module):
                    overlap = []
                    for drone_sequence in self.list_drone_feedback.keys():
                        if self.list_drone_feedback[drone_sequence][i] != -1:
                            overlap.append(drone_sequence)
                    if len(overlap) >= 2:  # a virtual overlap of transmission
                        self.schedule.append(self.rnd_mac.choice(list(set(overlap) | set(self.always_best))))
                    elif len(overlap) == 1:  # only one transmission
                        rv = self.rnd_mac.random()
                        if rv <= self.epsilon:  # random step with epsilon probability
                            self.schedule.append(self.rnd_mac.choice(self.always_best))
                        else:  # normal step with 1 - epsilon probability
                            self.schedule.append(overlap[0])
                    else:  # no transmission then we take one best drone uniform at random
                        self.schedule.append(self.rnd_mac.choice(self.always_best))

            self.schedule_counter += 1
            return self.simulator.drones[self.schedule[self.schedule_counter % self.module]]