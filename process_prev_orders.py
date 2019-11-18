import numpy as np
import jsonlines
from constants import COASTS, WATER, ORDERING, UNIT_TYPE, UNIT_POWER, ORDER_TYPE
from process import read_data, parse_states


def create_province_dict():
    '''
    Function to construct a dictionary of province names
    https://www.lspace.org/games/afpdip/files/abb.html
    '''
    provinces = {}
    for prov in ORDERING:
        provinces[prov] = {}

        # providing area type
        if prov in COASTS:
            provinces[prov]["area_type"] = "coast"
        elif prov in WATER:
            provinces[prov]["area_type"] = "water"
        else:
            provinces[prov]["area_type"] = "land"

    return provinces

def read_orders_data(orders,board_states):
    prev_orders_game = []
    for i in range(len(orders)):
        prev_orders = []
        for j in range(len(orders[i])):
            prev_orders.append(parse_order(orders[i][j],board_states[i][j]))
        prev_orders_game.append(prev_orders)
    return prev_orders_game

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
    return phase_matrices




if __name__=="__main__":
    states, orders, results = read_data("/media/daniel/DATA/diplomacy_data/standard_no_press.jsonl")
    board_states, season_names = parse_states(states)
    prev_orders = read_orders_data(orders,board_states)
    matrix = construct_prev_orders_matrix(prev_orders[1])
