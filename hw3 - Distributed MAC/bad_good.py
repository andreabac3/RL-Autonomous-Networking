from src.utilities import utilities as util
from src.utilities import config
from src.entities.uav_entities import Drone, DataPacket

"""
The class is responsable to allocate communication resources to neighbors drones that want to offload data to the depot.
We work over an semplified TDMA approach, each time step only one drone can receive the resource and communicate a packet to the depot. 
"""


class BAD_GOOD():

    def __init__(self, drone, simulator):
        self.simulator = simulator
        self.drone = drone
        self.print_stats = config.MAC_PRINT_STATS
        self.last_feedback = None

        # Simulation parameters
        self.len_simulation = self.simulator.len_simulation
        self.n_drones = self.simulator.n_drones

        # Hyper parameters
        self.epsilon = 0.05
        self.len_frame = self.n_drones * 2

        # Data structure for the action taken and for the Q_Table
        self.taken_action = {}  # id_packets -> time slot
        self.slot_score = {}  # {slot: {good:2, bad:3}

    def communicate(self, cur_step: int) -> bool:
        """ Return the True if the drone should communicate in this slot, False otherwise """
        # return True or False
        cur_slot = self._get_time_slot(cur_step)

        if self._get_exploration_step() or cur_slot not in self.slot_score:
            return self._get_bool_random()

        # Communicate if positive feedback are greater then negative
        if self.slot_score[cur_slot]['good'] > self.slot_score[cur_slot]['bad']:
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

        # We update the slot score w.r.t. positive/negative feedbacks
        if slot not in self.slot_score:
            self.slot_score[slot] = {'good': 1, 'bad': 0} if feedback else {'good': 0, 'bad': 1}
        else:
            if feedback:
                self.slot_score[slot]['good'] += 1
            else:
                self.slot_score[slot]['bad'] += 1

        if self.print_stats:
            print(packet, feedback)
        pass

    def run(self, cur_step: int):
        """ run the mac and allocate bandwidth to a particual drone """
        packets_to_send = self.drone.all_packets()  # the packets are ordered, from oldest to newest
        if len(packets_to_send) == 0:
            return

        cur_slot: int = self._get_time_slot(cur_step)
        communicate = self.communicate(cur_step)  # whether communicate or not

        # pck <- packets_to_send.pick_packet()
        # E.g., :
        # oldest_packet = packets_to_send[0]
        # newest_packet = packets_to_send[-1]

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