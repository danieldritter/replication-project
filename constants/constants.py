# importing from research
from diplomacy_research.models import state_space

# grabbing adjacency matrix as integers
A = state_space.get_adjacency_matrix("standard")

dLbo = 64
dLpo = 64
H_ENC_COLS = dLbo + dLpo
NUM_PLACES = 81 # note: 75 provinces + 6 coasts
ORDER_VOCABULARY_SIZE = 13042

# Predefined location order
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

# power types
ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

# province types
COASTS = ["BUL/EC", "BUL/SC", "SPA/NC", "SPA/SC", "STP/NC", "STP/SC"]
WATER = ["ADR", "AEG", "BAL", "BAR", "BLA", "EAS", "ENG", "BOT",
         "GOL", "HEL", "ION", "IRI", "MID", "NAT", "NTH", "NRG",
         "SKA", "TYN", "WES"]

# supply centers
OG_SUPPLY_CENTERS = {
    "AUSTRIA": ["BUD", "TRI", "VIE"],
    "ENGLAND": ["EDI", "LVP", "LON"],
    "FRANCE": ["BRE", "BUR", "MAR"],
    "GERMANY": ["BER", "KIE", "MUN"],
    "ITALY": ["NAP", "ROM", "VEN"],
    "RUSSIA": ["MOS", "SEV", "STP", "WAR"],
    "TURKEY": ["ANK", "CON", "SMY"],
    "NEUTRAL": ["NWY", "SWE", "DEN", "BEL", "HOL", "SPA", "POR",
                "TUN", "SER", "RUM", "BUL", "GRE"]
}

# mappings for constructing data for board states
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

BUILDABLE_REMOVABLE = {
    "buildable": [1, 0],
    "removable": [0, 1]
}

AREA_TYPE = {
    "land": [1, 0, 0],
    "water": [0, 1, 0],
    "coast": [0, 0, 1]
}

# mappings for constructing data for previous orders
ORDER_TYPE = {
    "H": [1, 0, 0, 0, 0],
    "S": [0, 1, 0, 0, 0],
    "C": [0, 0, 1, 0, 0],
    "-": [0, 0, 0, 1, 0],
    None: [0, 0, 0, 0, 1]
}

SEASON = {
    "W": [1,0,0,0],
    "F": [0,1,0,0],
    "S": [0,0,1,0],
    "C": [0,0,0,1]
}
