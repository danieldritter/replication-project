import numpy as np
import jsonlines
from constants import COASTS, WATER, ORDERING, OG_SUPPLY_CENTERS


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
            if count == 10:
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

    # format structure [province 1 (7 elements), province 2 (7 elems ...)]
    for i in range(len(states)):
        game_data = []
        game_seasons = []
        phases = states[i]

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
                                    province_dict[prov]["buildable_removable"] = "buildable"

                # make provinces with units removable
                elif power_builds["count"] == -1:
                    for prov in province_dict:
                        if province_dict[prov] == power:
                            if province_dict[prov]["unit_type"] != None:
                                province_dict[prov]["buildable_removable"] = "removable"
            game_data.append(province_dict)
        board_dict_list.append(game_data)
        season_names.append(game_seasons)

    return board_dict_list, season_names

# def construct_state_matrix(board_dict_list):
#     '''
#     Function to create the matrix inputs to the encoder model

#     Keyword Args:
#     board_dict_list - a list of board dictionaries 

#     Returns:
#     a list of matrices that are ordered by our specificed ORDERING matrix
#     '''

#     board_matrix_data = []

#     for phase in board_dict_list:
#         phase_data = []
#         for province in ORDERING:


if __name__ == "__main__":
    states, orders, results = read_data("data/standard_no_press.jsonl")
    board_dict_list, season_names = parse_states(states)

    for game in board_dict_list:
        print(len(game))