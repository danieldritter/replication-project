import numpy as np
import jsonlines
from constants.constants import COASTS, WATER, ORDERING, OG_SUPPLY_CENTERS, UNIT_TYPE, UNIT_POWER, AREA_TYPE, ORDER_TYPE, NUM_POWERS
from RL.reward import Reward
def create_province_dict():
    '''
    Function to construct a dictionary of province names
    https://www.lspace.org/games/afpdip/files/abb.html
    '''

    province_dict = {}

    for prov in ORDERING:
        province_dict[prov] = {}

        # providing area type
        if prov in COASTS:
            province_dict[prov]["area_type"] = "coast"
        elif prov in WATER:
            province_dict[prov]["area_type"] = "water"
        else:
            province_dict[prov]["area_type"] = "land"

    return province_dict


def read_data(filepath):
    '''
    Function to read the json data

    Keyword Args:
    filepath - the file to read from

    Returns:
    an arry of states, orders, and results from the json
    '''

    states, orders, results = [], [], []
    count = 0
    # TODO: Probably need to separate by turn here
    with jsonlines.open(filepath) as file:
        for game in file:
            phase_states = []
            phase_orders = []
            phase_results = []
            for phase in game["phases"]:
                phase_states.append(phase["state"])
                phase_orders.append(phase["orders"])
                phase_results.append(phase["results"])
            if count == 20:
                break
            count += 1
            states.append(phase_states)
            orders.append(phase_orders)
            results.append(phase_results)

    return states, orders, results

def parse_states(states):
    '''
    Function to parse the board state information

    Keyword Args:
    the states list passed from read_data

    Returns:
    a dictionary containing the board states of each phase and a list of the
    season names

    --------------------------------------------------------------------------
    Data Format:
    Each game is a dictionary of ["id", "map", "rules", "phases"]

    We only really consider phases:
    Phases is a list of dictionaries where each dictionary has
    ["name", "state", "orders", "results", "messages"]

    "name" has Season (F, W, or S) - Year (0000) - Phase (M (movement), A (adjustment),
    R (retreat))
    "state" dictionary of dictionaries containing
    --------------------------------------------------------------------------
    '''

    season_names = []
    board_dict_list = []
    supply_center_owners = []

    # format structure [province 1 (7 elements), province 2 (7 elems ...)]
    for i in range(len(states)):
        game_data = []
        game_seasons = []
        phases = states[i]
        supply_center_owners_per_phase = []
        # looping through phases of a game
        for s in phases:
            # extracting seas on information for FiLM
            game_seasons.append(s["name"])

            # global dictionary for province names
            province_dict = create_province_dict()

            # adding unit type andbuilds owner of unit
            units = s["units"]
            for power in units:
                result = units[power]
                for r in result:
                    # if r.startswith("*"):
                        # print(r)
                    type, province = r.split()
                    province_dict[province]["unit_type"] = type
                    province_dict[province]["unit_power"] = power

            # adding in owner of supply center
            centers = s["centers"]
            for power in centers:
                result = centers[power]
                for r in result:
                    province_dict[province]["supply_center_owner"] = power

                    # adding unit type
                    province_dict[province]["unit_type"] = type
                    province_dict[province]["unit_power"] = power

            # adding dislodged information
            retreats = s["retreats"]
            for power in retreats:
                result = retreats[power]
                for unit in result:
                    type, province = unit.split()
                    province_dict[province]["d_unit_type"] = type
                    province_dict[province]["d_unit_power"] = power

            # adding buildable/removable ??? what is this!
            for power_name in s["builds"]:
                power = power_name[:3]
                power_builds = s["builds"][power_name]
                # make supply centers buildable
                if power_builds["count"] == 1:
                    for prov in province_dict:
                        # checking if a supply centers
                        if "supply_center_owner" in province_dict[prov]:
                            sc_owner = province_dict[prov]["supply_center_owner"]
                            # checking for correct power
                            if sc_owner == power:
                                print("hit")
                                # only for original supply centers
                                if prov in  OG_SUPPLY_CENTERS[sc_owner]:
                                    province_dict[prov]["buildable"] = 1

                # make provinces with units removable
                if power_builds["count"] == -1:
                    for prov in province_dict:
                        if province_dict[prov] == power:
                            if province_dict[prov]["unit_type"] != None:
                                province_dict[prov]["removable"] = 1
            game_data.append(province_dict)
            supply_center_owners_per_phase.append(centers)
        board_dict_list.append(game_data)
        season_names.append(game_seasons)
        supply_center_owners.append(supply_center_owners_per_phase)

    return board_dict_list, season_names, supply_center_owners

def construct_state_matrix(board_state_game):
    '''
    Function to create the matrix inputs to the encoder model

    Keyword Args:
    board_state_game - a game containing a list of phases, which are a dictionary of board_state data

    Returns:
    a matrix that contains
    '''

    category_order = [
        "unit_type", "unit_power", "buildable_removable",
        "d_unit_type", "d_unit_power", "area_type", "supply_center_owner"
    ]

    board_matrix_data = []

    for phase in board_state_game:
        phase_data = np.zeros((81, 35))

        # getting data for province
        for i in range(len(ORDERING)):
            province = ORDERING[i]
            if "unit_type" in phase[province]:

                # checking to remove * in front
                unit_type = phase[province]["unit_type"]
                if unit_type.startswith("*"):
                    unit_type = unit_type[1:]
                unit_type = UNIT_TYPE[unit_type]
            else:
                unit_type = UNIT_TYPE[None]

            if "unit_power" in phase[province]:
                unit_power = UNIT_POWER[phase[province]["unit_power"]]
            else:
                unit_power = UNIT_POWER[None]

            if "buildable" in phase[province]:
                b = phase[province]["buildable"]
            else:
                b = [0]

            if "buildable" in phase[province]:
                r = phase[province]["removable"]
            else:
                r = [0]

            if "d_unit_type" in phase[province]:
                d_unit_type = UNIT_TYPE[phase[province]["d_unit_type"]]
            else:
                d_unit_type = UNIT_TYPE[None]

            if "d_unit_power" in phase[province]:
                d_unit_power = UNIT_POWER[phase[province]["d_unit_power"]]
            else:
                d_unit_power = UNIT_POWER[None]

            if "area_type" in phase[province]:
                area_type = AREA_TYPE[phase[province]["area_type"]]

            if "supply_center_owner" in phase[province]:
                supply_center_owner = UNIT_POWER[phase[province]["supply_center_owner"]]
            else:
                supply_center_owner = UNIT_POWER[None]

            phase_data[i] = unit_type + unit_power + b + r + d_unit_type + d_unit_power + area_type + supply_center_owner
        board_matrix_data.append(phase_data)
    return np.array(board_matrix_data)

def read_orders_data(orders,board_states):
    prev_orders_game = []
    prev_orders_game_labels = []
    for i in range(len(orders)):
        prev_orders = []
        prev_orders_labels = []
        for j in range(len(orders[i])):
            prev_orders.append(parse_order(orders[i][j],board_states[i][j]))
            prev_orders_labels.append(orders[i][j])
        prev_orders_game.append(prev_orders)
        prev_orders_game_labels.append(prev_orders_labels)
    return prev_orders_game, prev_orders_game_labels

def parse_order(orders,board_state):
    # How do we handle case where unit_type == None
    turn_dict = create_province_dict()
    for power in orders.keys():
        if orders[power] != None:
            for order in orders[power]:
                order_components = order.split()
                province = order_components[1]
                if order_components[2] in ["H","S","C","-"]:
                    turn_dict[province]["unit_type"] = order_components[0]
                    # TODO: Might be none in some cases, so need to account for that
                    if "supply_center_owner" in board_state[province].keys():
                        turn_dict[province]["supply_center_owner"] = board_state[province]["supply_center_owner"]

                    turn_dict[province]["order_type"] = order_components[2]
                    turn_dict[province]["issuing_power"] = board_state[province]["unit_power"]
                # Hold Case
                if order_components[2] == "H":
                    turn_dict[province]["source_power"] = None
                    # Dest power might be wrong
                    turn_dict[province]["dest_power"] = None
                # Support Case
                elif order_components[2] == "S":
                    if len(order_components) == 5:
                        turn_dict[province]["source_power"] = board_state[order_components[4]]["unit_power"]
                        turn_dict[province]["dest_power"] = None
                    elif len(order_components) == 7:
                        turn_dict[province]["source_power"] = board_state[order_components[4]]["unit_power"]
                        if "unit_power" in board_state[order_components[6]].keys():
                            turn_dict[province]["dest_power"] = board_state[order_components[6]]["unit_power"]
                # Move Case
                elif order_components[2] == "-":
                    turn_dict[province]["source_power"] = None
                    if "unit_power" in board_state[order_components[3]].keys():
                        turn_dict[province]["dest_power"] = board_state[order_components[3]]["unit_power"]
                # Convoy Case
                elif order_components[2] == "C":
                    turn_dict[province]["source_power"] = board_state[order_components[4]]["unit_power"]
                    if "unit_power" in board_state[order_components[6]].keys():
                        turn_dict[province]["dest_power"] = board_state[order_components[6]]["unit_power"]
                # Build Case
                elif order_components[2] == "B":
                    continue
                # Disband Case
                elif order_components[2] == "D":
                    continue
                # Retreat Case
                elif order_components[2] == "R":
                    continue
                else:
                    print("Problem")
                    print(order_components[2])
        return turn_dict

def construct_prev_orders_matrix(prev_orders_game_state):
    phase_matrices = []
    for phase in prev_orders_game_state:
        matrix = np.zeros((81,40))
        for i,prov in enumerate(ORDERING):
            if "unit_type" in phase[prov].keys():
                row_part1 = UNIT_TYPE[phase[prov]["unit_type"]]
            else:
                row_part1 = UNIT_TYPE[None]
            if "issuing_power" in phase[prov].keys():
                row_part2 = UNIT_POWER[phase[prov]["issuing_power"]]
            else:
                row_part2 = UNIT_POWER[None]
            if "order_type" in phase[prov].keys():
                row_part3 = ORDER_TYPE[phase[prov]["order_type"]]
            else:
                row_part3 = ORDER_TYPE[None]
            if "source_power" in phase[prov].keys():
                row_part4 = UNIT_POWER[phase[prov]["source_power"]]
            else:
                row_part4 = UNIT_POWER[None]
            if "dest_power" in phase[prov].keys():
                row_part5 = UNIT_POWER[phase[prov]["dest_power"]]
            else:
                row_part5 = UNIT_POWER[None]
            if "supply_center_owner" in phase[prov].keys():
                row_part6 = UNIT_POWER[phase[prov]["supply_center_owner"]]
            else:
                row_part6 = UNIT_POWER[None]
            matrix[i] = row_part1 + row_part2 + row_part3 + row_part4 + row_part5 + row_part6
        phase_matrices.append(matrix)
    return np.array(phase_matrices)

def get_data(filepath):
    """


    :param filepath:
    :return:
    """
    states, orders, results = read_data(filepath)
    board_dict_list, season_names, supply_center_owners = parse_states(states)
    prev_orders, prev_orders_game_labels = read_orders_data(orders, board_dict_list)
    # print('ORDERS LIST: ', prev_orders)
    state_inputs = np.array([construct_state_matrix(game) for game in board_dict_list])
    prev_order_inputs = np.array([construct_prev_orders_matrix(game) for game in prev_orders])
    return state_inputs, prev_order_inputs, prev_orders_game_labels, season_names, supply_center_owners, board_dict_list

if __name__ == "__main__":
    states, orders, results = read_data("standard_no_press.jsonl")
    board_dict_list, season_names = parse_states(states)
    get_returns(board_dict_list)
