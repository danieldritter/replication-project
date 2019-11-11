import numpy as np
import jsonlines
from test import ordering
from constants import COASTS, WATER


def create_province_dict():
    '''
    Function to construct a dictionary of province names
    https://www.lspace.org/games/afpdip/files/abb.html
    '''

    for i in ordering:
        prov = p_lines[i]
        provinces[prov] = {}

        # providing area type
        if prov in COASTS:
            provinces[prov]["area_type"] = "coast"
        elif prov in WATER:
            provinces[prov]["area_type"] = "water"
        else:
            provinces[prov]["area_type"] = "land"

    return provinces


def read_data(filepath):
    '''
    Function to read the json data

    Data Format:
    Each game is a dictionary of ["id", "map", "rules", "phases"]

    We only really consider phases:
    Phases is a list of dictionaries where each dictionary has ["name", "state", "orders", "results", "messages"]

    "name" has Season (F, W, or S) - Year (0000) - Phase (M (movement), A (adjustment), R (retreat))
    "state" dictionary of dictionaries containing
    '''

    states, orders, results = [], [], []
    season_names = []
    count = 0


    with jsonlines.open(filepath) as file:
        for game in file:
            for phase in game["phases"]:
                states.append(phase["state"])
                orders.append(phase["orders"])
                results.append(phase["results"])
            if count == 10:
                break
            count += 1

    # format structure [province 1 (7 elements), province 2 (7 elems ...)]
    for i in range(len(states)):
        s = states[i]

        # extracting seas on information for FiLM
        season_names.append(s["name"])

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
            print(result)
            for unit in result:
                print(unit)
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
                    if province_dict[prov]["supply_center_owner"] == power:
                        province_dict[prov]["buildable_removable"] = "buildable"
            # make provinces with units removable
            elif power_builds["count"] == -1:
                for prov in province_dict:
                    if province_dict[prov] == power:
                        if province_dict[prov]["unit_type"] != None:
                            province_dict[prov]["buildable_removable"] = "removable"
            
    return states, orders, results, season_names, province_dict


if __name__ == "__main__":
    s, o, r, seasons, provinces_result = read_data("data/standard_no_press.jsonl")
