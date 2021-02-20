from src.entities.uav_entities import Drone
from src.mac_protocol.depot_mac import DepotMAC
import numpy as np

"""
The class is responsable to allocate communication resources to neighbors drones that want to offload data to the depot.
We work over an semplified TDMA approach, each time step only one drone can receive the resource and communicate a packet to the depot. 
"""


class AIncremental(DepotMAC):

    def __init__(self, simulator, depot):
        super().__init__(simulator, depot)
        self.rnd_mac = np.random.RandomState(self.simulator.seed)
        self.N = [1] * simulator.n_drones
        self.Q = [9] * simulator.n_drones  # optimistic initial value --- (best ones)
        self.epsilon = 0.2  # epsilon variable

    def allocate_resource_to_drone(self, drones: list, cur_step: int) -> Drone:
        """ Return the drone to who allocate bandwith for upload data in this step """

        if self.last_feedback != None:
            (drone, transmission, feedback) = self.last_feedback

        if cur_step == 0:
            return self.rnd_mac.choice(drones)  # at the first step we choose a random drone

        # the follow sequence of "if" is used to define the reward
        if transmission == True and feedback > 0:
            reward = 1 + feedback
        elif transmission == True and feedback == 0:
            reward = 1
        elif transmission == False and feedback > 0:
            reward = feedback
        elif transmission == False and feedback == 0:
            reward = 0

        # we apply the incremental update rule
        self.Q[drone.identifier] = self.Q[drone.identifier] + (1 / self.N[drone.identifier]) * (
                    reward - self.Q[drone.identifier])
        rv = self.rnd_mac.rand()

        if rv <= self.epsilon:  # random step with probability epsilon
            drone_to_return = self.rnd_mac.choice(drones)  # choose a random drone

        else:  # greedy step with probability 1 - epsilon

            max_value = max(self.Q)  # search the best value in the Qtable
            indices = [index for index, value in enumerate(self.Q) if value == max_value]
            # take all the indices that have that best value (max_value)
            drone_id = self.rnd_mac.choice(indices)  # choose a random index from the "best" ones
            drone_to_return = drones[drone_id]  # define the drone to return using the index id

        self.N[drone_to_return.identifier] += 1  # increment the counter that tell how many time we pick that drone

        return drone_to_return