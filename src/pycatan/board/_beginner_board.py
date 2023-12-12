import random
from ._board import Board
from ._hex import Hex
from ._coords import Coords
from ._hex_type import HexType
from ._harbor import Harbor
from .._resource import Resource


class BeginnerBoard(Board):
    """The beginner board, as outlined in the Catan rules."""

    def __init__(self):
        hexes = self.generate_random_hexes()

        super().__init__(
            hexes=hexes,
            harbors=[
                Harbor(path_coords={Coords(4, 0), Coords(3, 1)}, resource=None),
                Harbor(path_coords={Coords(1, 3), Coords(0, 4)}, resource=None),
                Harbor(path_coords={Coords(-2, 5), Coords(-3, 5)}, resource=None),
                Harbor(path_coords={Coords(-4, 3), Coords(-4, 4)}, resource=None),
                Harbor(path_coords={Coords(-4, 0), Coords(-4, 1)}, resource=None),
                Harbor(path_coords={Coords(-2, -3), Coords(-3, -2)}, resource=None),
                Harbor(path_coords={Coords(2, -5), Coords(1, -4)}, resource=None),
                Harbor(path_coords={Coords(3, -4), Coords(4, -4)}, resource=None),
                Harbor(path_coords={Coords(5, -3), Coords(5, -2)}, resource=None),
            ])

    def generate_random_hexes(self):
        # Define the desired distribution of resource types and token numbers (excluding desert)
        resource_distribution = [
            HexType.MOUNTAINS, HexType.PASTURE, HexType.FOREST,
            HexType.FIELDS, HexType.HILLS, HexType.PASTURE,
            HexType.HILLS, HexType.FIELDS, HexType.FOREST,
            HexType.FOREST, HexType.MOUNTAINS, HexType.FOREST,
            HexType.MOUNTAINS, HexType.FIELDS, HexType.PASTURE,
            HexType.HILLS, HexType.FIELDS, HexType.PASTURE
        ]

        token_number_distribution = [
            2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12
        ]

        # Shuffle the distributions
        random.shuffle(resource_distribution)
        random.shuffle(token_number_distribution)

        # Create a list of hex coordinates
        hex_coordinates = [
            Coords(4, -2),
            Coords(3, 0),
            Coords(2, 2),
            Coords(3, -3),
            Coords(2, -1),
            Coords(1, 1),
            Coords(0, 3),
            Coords(2, -4),
            Coords(1, -2),
            Coords(-1, 2),
            Coords(-2, 4),
            Coords(0, -3),
            Coords(-1, -1),
            Coords(-2, 1),
            Coords(-3, 3),
            Coords(-2, -2),
            Coords(-3, 0),
            Coords(-4, 2),
            Coords(0,0)
        ]

        # Shuffle the list of coordinates
        random.shuffle(hex_coordinates)

        hexes = []

        # Assign the desert hex to the first coordinate in the shuffled list
        desert_hex = hex_coordinates.pop(0)
        hexes.append(Hex(desert_hex, HexType.DESERT, None))

        # Distribute the remaining resource types and token numbers among the shuffled coordinates
        for hex_coord in hex_coordinates:
            hex_type = resource_distribution.pop()
            token_number = token_number_distribution.pop()
            hexes.append(Hex(hex_coord, hex_type, token_number))

        return hexes
