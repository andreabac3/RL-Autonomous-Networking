from src.utilities import utilities as util
from src.utilities import config
from src.entities.uav_entities import Drone, DataPacket

"""
The class is responsable to allocate communication resources to neighbors drones that want to offload data to the depot.
We work over an semplified TDMA approach, each time step only one drone can receive the resource and communicate a packet to the depot. 
"""

class QL():

    def __init__(self, drone, simulator):
        self.simulator = simulator
        self.drone = drone
        self.print_stats = config.MAC_PRINT_STATS
        self.last_feedback = None

        # Simulation parameters
        self.len_simulation = self.simulator.len_simulation
        self.n_drones = self.simulator.n_drones

        # Hyper parameters
        self.alpha = 0.1
        self.epsilon = 0.05
        self.len_frame = self.n_drones * 2

        # Data structure for the action taken and for the Q_Table
        self.taken_action = {}  # id_packets -> time slot
        self.q_table = {}


    def communicate(self, cur_step: int) -> bool:
        """ Return the True if the drone should communicate in this slot, False otherwise """

        cur_slot = self._get_time_slot(cur_step)

        if self._get_exploration_step():
            return self._get_bool_random()

        # The drone communicate according to the following threshold
        if self.q_table[cur_slot] >= 0:
            return True

        return False

    def feedback(self, feedback: bool, packet):
        """
        The method is called automatically to notify the status of tha last packet delivered, for simplicity
            we add also the referred packet in the feedback.

        :param feedback: True if last packet was delivery succesfully, False otherwise
        :param packet: the referred packet
        :return:
        """
        id_packet: int = packet.identifier
        slot: int = self.taken_action[id_packet]

        reward: int = 1 if feedback else -1

        # We update the Q_table with the following ALOHA-Q formula
        self.q_table[slot] += self.alpha * (reward - self.q_table[slot])

        if self.print_stats:
            print(packet, feedback)
        pass

    def run(self, cur_step: int):
        """ run the mac and allocate bandwidth to a particual drone """
        packets_to_send = self.drone.all_packets()  # the packets are ordered, from oldest to newest
        if len(packets_to_send) == 0:
            return

        cur_slot: int = self._get_time_slot(cur_step)
        if cur_slot not in self.q_table:
            self.q_table[cur_slot] = 0

        communicate = self.communicate(cur_step)  # whether communicate or not

        # pck <- packets_to_send.pick_packet()
        # E.g., :
        # oldest_packet = packets_to_send[0]
        # newest_packet = packets_to_send[-1]

        # We select the oldest packet
        oldest_packet: DataPacket = packets_to_send[0]

        if communicate:

            id_packet: int = oldest_packet.identifier
            self.taken_action[id_packet] = cur_slot

            self.simulator.depot.receive(self.drone, oldest_packet)
            if self.print_stats:
                print("Transmission for drone: ", self.drone.identifier, " to depot, ",
                      len(packets_to_send), " in the buffer.")

    # Function that identify the current slot
    def _get_time_slot(self, cur_step: int) -> int:
        return cur_step % self.len_frame

    # Function that define the exploration/exploitation step
    def _get_exploration_step(self) -> bool:
        rv = self.simulator.rnd_routing.random()
        return rv < self.epsilon

    # Return True or False u.a.r.
    def _get_bool_random(self) -> bool:
        return self.simulator.rnd_routing.random() > 0.5