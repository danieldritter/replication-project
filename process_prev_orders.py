import numpy as np
import jsonlines
from constants import COASTS, WATER, ORDERING
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
    prev_orders = []
    for i in range(len(orders)):
        prev_orders.append(parse_order(orders[i],board_states[i]))

def parse_order(orders,board_state):
    # How do we handle case where unit_type == None
    turn_dict = create_province_dict()
    for power in orders.keys():
        for order in orders[power]:
            order_components = order.split()
            print(order_components)
            province = order_components[1]
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
                return turn_dict
            # Move Case
            elif order_components[2] == "-":
                turn_dict[province]["source_power"] = None
                if "unit_power" in board_state[order_components[3]].keys():
                    turn_dict[province]["dest_power"] = board_state[order_components[3]]["unit_power"]
                return turn_dict
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



if __name__=="__main__":
    states, orders, results = read_data("/media/daniel/DATA/diplomacy_data/standard_no_press.jsonl")
    board_states, season_names = parse_states(states)
    prev_orders = read_orders_data(orders,board_states)
    print(prev_orders)
