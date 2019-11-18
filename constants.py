# For constants
A = []
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
    "A": 0,
    "F": 1,
    None: 2
}

UNIT_POWER = {
    "AUSTRIA": 0,
    "ENGLAND": 1,
    "FRANCE": 2,
    "GERMANY": 3,
    "ITALY": 4,
    "RUSSIA": 5,
    "TURKEY": 6,
    None: 7
}

BUILDABLE_REMOVABLE = {
    "buildable": 0,
    "removable": 1
}

AREA_TYPE = {
    "land": 0,
    "water": 1, 
    "coast": 2
}

# mappings for constructing data for previous orders
ORDER_TYPE = {
    "H": 0,
    "S": 1,
    "C": 2,
    "-": 3,
    None: 4
}