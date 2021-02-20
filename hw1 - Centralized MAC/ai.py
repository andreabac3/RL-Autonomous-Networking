from src.entities.uav_entities import Drone
from src.mac_protocol.depot_mac import DepotMAC
import numpy as np

"""
The class is responsable to allocate communication resources to neighbors drones that want to offload data to the depot.
We work over an semplified TDMA approach, each time step only one drone can receive the resource and communicate a packet to the depot. 
"""


class AI_MAC(DepotMAC):

    def __init__(self, simulator, depot):
        super().__init__(simulator, depot)
        self.rnd_mac = np.random.RandomState(self.simulator.seed)

        # list used to count the number of packets generated from each drone
        self.packets = [0] * self.simulator.n_drones
        # list used to represent the probability that a drone can generate a packet
        self.probs = [0] * self.simulator.n_drones
        self.N = [1] * self.simulator.n_drones  # list used to count the number of time that a drone is queried
        self.Q = [0] * self.simulator.n_drones  # list used to define the estimated reward value
        self.epsilon = 0.05  # epsilon best value

    def allocate_resource_to_drone(self, drones: list, cur_step: int) -> Drone:
        """ Return the drone to who allocate bandwith for upload data in this step """

        if self.last_feedback != None:
            (drone, transmission, feedback) = self.last_feedback
            self.packets[drone.identifier] += feedback

        # the following is the learning phase
        if cur_step < 0.17 * self.simulator.len_simulation:
            drone_to_return = self.rnd_mac.choice(drones)
            self.probs = [(self.packets[d.identifier] / (sum(self.packets) + 1)) for d in drones]
            return drone_to_return

        else:  # after the learning steps
            self.Q = [(self.probs[d.identifier] * 100) + 1 for d in drones]

            # how we update the estimated reward values
            if transmission and feedback > 0:
                self.Q[drone.identifier] += 1 + feedback
            elif transmission and feedback == 0:
                self.Q[drone.identifier] += 0
            elif transmission == False and feedback > 0:
                self.Q[drone.identifier] += feedback
            elif transmission == False and feedback == 0:
                self.Q[drone.identifier] /= 2

        rv = self.rnd_mac.rand()

        if rv < self.epsilon:  # random step chosen with epsilon probability
            drone_to_return = self.rnd_mac.choice(drones)

        else:  # greedy step chosen with probability 1 - epsilon

            max_value = max(self.Q)
            indices = [index for index, value in enumerate(self.Q) if value == max_value]
            """
            indices is the index list of the best drones w.r.t. max Q value
            to avoid to choose always the same drone from indices we pick one uniform at random  
            """
            drone_id = self.rnd_mac.choice(indices)
            drone_to_return = drones[drone_id]

        self.N[drone_to_return.identifier] += 1

        return drone_to_return

