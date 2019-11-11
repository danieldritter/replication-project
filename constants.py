# For constants
A = []
dLbo = 64
dLpo = 64
H_ENC_COLS = dLbo + dLpo
NUM_PLACES = 81 # note: 75 provinces + 6 coasts

# Predefined location order
STANDARD_TOPO_LOCS = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
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

# province types
COASTS = ["BUL/EC", "BUL/SC", "SPA/NC", "SPA/SC", "STP/NC", "STP/SC"]
WATER = ["ADR", "AEG", "BAL", "BAR", "BLA", "EAS", "ENG", "BOT", 
         "GOL", "HEL", "ION", "IRI", "MID", "NAT", "NTH", "NRG", 
         "SKA", "TYN", "WES"]