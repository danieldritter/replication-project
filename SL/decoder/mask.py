import sys
import numpy as np
import tensorflow as tf
from diplomacy import Game
from diplomacy_research.models import state_space
# from constants.constants import ORDER_VOCABULARY_SIZE, ORDERING, UNIT_TYPE, UNIT_POWER
ORDER_VOCABULARY_SIZE = 13042
ORDERING = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
                      'NWG', 'ENG', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
                      'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC',
                      'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
                      'STP/NC', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', 'SPA/NC',
                      'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'BOT', 'LVN',
                      'PRU', 'STP/SC', 'MOS', 'TUN', 'LYO', 'TYS', 'PIE',
                      'BOH', 'SIL', 'TYR', 'WAR', 'SEV', 'UKR', 'ION',
                      'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
                      'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU',
                      'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR',
                      'BUL', 'BUL/EC', 'CON', 'BUL/SC']
UNIT_TYPE = {
    "A": [1, 0, 0],
    "F": [0, 1, 0],
    None:[0, 0, 1]
}

UNIT_POWER = {
    "AUSTRIA": [1, 0, 0, 0, 0, 0, 0, 0],
    "ENGLAND": [0, 1, 0, 0, 0, 0, 0, 0],
    "FRANCE": [0, 0, 1, 0, 0, 0, 0, 0],
    "GERMANY": [0, 0, 0, 1, 0, 0, 0, 0],
    "ITALY": [0, 0, 0, 0, 1, 0, 0, 0],
    "RUSSIA": [0, 0, 0, 0, 0, 1, 0, 0],
    "TURKEY": [0, 0, 0, 0, 0, 0, 1, 0],
    None: [0, 0, 0, 0, 0, 0, 0, 1]
}


def masked_softmax(arr, mask):
    '''
    Masks out invalid orders using the given mask and performs softmax over remaining

    Args:
    arr - array to mask and softmax
    mask - binary mask of the same size as arr.

    Returns:
    Softmax'ed result of applying given mask to h_dec
    '''
    return tf.nn.softmax(tf.math.multiply(arr, mask))

def create_mask(board_state, phase, loc):
    '''
    Given a board_state, produces a mask that only includes valid orders from loc,
    based on their positions in get_order_vocabulary in state_space. Assumes playing on standard map

    Args:
    board_state - 81 x dilbo vector representing current board state as described in Figure 2
    phase - string indicating phase of game (e.g. 'S1901M')
    loc - string representing location (e.g. 'PAR')

    Returns:
    Mask for zeroing out invalid orders, length is the number of orders total
    '''

    # create instance of Game object based on board_state
    game = Game(map_name='standard')

    game.set_current_phase(phase)
    game.clear_units()

    power_units = {}
    power_centers = {}

    for loc_idx in range(len(board_state)):
        loc_name = ORDERING[loc_idx]
        loc_vec = board_state[loc_idx,:]

        # extract one hot vectors from encoding
        unit_type_one_hot = loc_vec[0:3]
        unit_power_one_hot = loc_vec[3:11]
        buildable = loc_vec[11]
        removable = loc_vec[12]
        dislodged_unit_type_one_hot = loc_vec[13:16]
        dislodged_unit_power_one_hot = loc_vec[16:24]
        area_type_one_hot = loc_vec[24:27]
        supply_center_owner_one_hot = loc_vec[27:35]

        # convert one hot vectors into indices, and index into unit types and powers
        unit_type = list(UNIT_TYPE)[np.argmax(unit_type_one_hot)]
        unit_power = list(UNIT_POWER)[np.argmax(unit_power_one_hot)]
        dislodged_unit_type = list(UNIT_TYPE)[np.argmax(dislodged_unit_type_one_hot)]
        dislodged_unit_power = list(UNIT_POWER)[np.argmax(dislodged_unit_power_one_hot)]
        supply_center_owner = list(UNIT_POWER)[np.argmax(supply_center_owner_one_hot)]

        # add the unit and/or dislodged unit in this locatino to power_units dict
        # likewise for supply center (if it exists). See set_units() documentation for how units are formatted
        if unit_type != None:
            if unit_power not in power_units:
                power_units[unit_power] = []
            power_units[unit_power].append('{} {}'.format(unit_type, loc_name))
        if dislodged_unit_type != None:
            if dislodged_unit_type not in power_units:
                power_units[dislodged_unit_power] = []
            power_units[dislodged_unit_power].append('*{} {}'.format(dislodged_unit_type, loc_name))
        if supply_center_owner != None:
            if supply_center_owner not in power_centers:
                power_centers[supply_center_owner] = []
            power_centers[supply_center_owner].append(loc_name)

    # Setting units
    game.clear_units()
    for power_name in list(power_units.keys()):
        game.set_units(power_name, power_units[power_name])

    # Setting centers
    game.clear_centers()
    for power_name in list(power_centers.keys()):
        game.set_centers(power_name, power_centers[power_name])

    possible_orders_at_loc = game.get_all_possible_orders()[loc]
    mask_arr = np.zeros(ORDER_VOCABULARY_SIZE)

    for order in possible_orders_at_loc:
        ix = state_space.order_to_ix(order)
        mask_arr[ix] = 1
    # return tf.convert_to_tensor(mask_arr, dtype=tf.float32)
    return mask_arr

def test_create_mask():
    arr = np.zeros((81,35))
    arr[:,0] = 1
    arr[:,3] = 1
    arr[:,24] = 1
    arr[:27] = 1
    mask = create_mask(arr, 'S1901M', 'PAR')
    print(mask)

if __name__ == "__main__":
    test_create_mask()
