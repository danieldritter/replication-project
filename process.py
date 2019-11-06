import numpy as np
import jsonlines


def create_province_names():
    '''
    Function to construct a dictionary of province names
    https://www.lspace.org/games/afpdip/files/abb.html
    '''

    provinces = {}
    with open("province_titles.txt") as f:
        p_lines = f.readlines()

    with open("province_types.txt") as g:
        t_lines = g.readlines()

    for i in range(len(p_lines)):
        prov = p_lines[i].upper().strip()
        provinces[prov] = {}

        # providing supply center owners
        t = t_lines[i]
        if t == "A":
            provinces[prov]["sc_owner"] = "AUS"
        elif t == "E":
            provinces[prov]["sc_owner"] = "ENG"
        elif t == "F":
            provinces[prov]["sc_owner"] = "FRA"
        elif t == "G":
            provinces[prov]["sc_owner"] = "GER"
        elif t == "I":
            provinces[prov]["sc_owner"] = "ITA"
        elif t == "R":
            provinces[prov]["sc_owner"] = "RUS"
        elif t == "T":
            provinces[prov]["sc_owner"] = "TUR"
        else:
            provinces[prov]["sc_owner"] = None

        # providing area type
        if t == "l":
            provinces[prov]["area_type"] = "Land"
        elif t == "w":
            provinces[prov]["area_type"] = "Water"
        # idk if this is actually correct
        elif t == "x":
            provinces[prov]["area_type"] = "Coast"
        else:
            provinces[prov]["area_type"] = None

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
    seasons = []
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
    for s in states:

        # extracting seas on information for FiLM
        seasons.append(s["name"])

        # global dictionary for province names
        province_names = create_province_names()

        # adding unit type and owner of unit
        units = s["units"]
        for power in units:
            result = units[power]
            for r in result:
                type, province = r.split()
                province_names[province]["unit_type"] = type
                province_names[province]["unit_power"] = power

        # adding in owner of supply center
        centers = s["centers"]
        for power in centers:
            result = centers[power]
            for r in result:
                province_names[province]["supply_center_owner"] = power

                # adding unit type
                province_names[province]["_unit_type"] = type
                province_names[province]["unit_power"] = power

        # adding dislodged information
        retreats = s["retreats"]
        for power in retreats:
            result = retreats[power]
            for r in result:
                for unit in r:
                    print(unit)
                    # type, province = unit.split()
                    # province_names[province]["d_unit_type"] = type
                    # province_names[province]["d_unit_power"] = power


        # adding buildable/removable ??? what is this!

    return states, orders, results, seasons, province_names


if __name__ == "__main__":
    s, o, r, seasons, provinces_result = read_data("/media/daniel/DATA/diplomacy_data/standard_no_press.jsonl")
