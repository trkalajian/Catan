from colored import stylize, fore_rgb, back_rgb
# from termcolor import colored
from typing import Optional, Dict, Tuple

from . import _board
from ._coords import Coords
from ._intersection import Intersection
from ._path import Path
from .._player import Player
from ._hex_type import HexType
from ._building_type import BuildingType
from ._hex import Hex
from .._resource import Resource


class BoardRenderer:
    DEFAULT_PLAYER_COLORS = [(0, 196, 13), (255, 0, 217), (0, 0, 255), (0, 255, 255)]

    DEFAULT_HEX_COLORS = {
        HexType.FIELDS: (255, 234, 41),  # RGB for "#ffea29"
        HexType.FOREST: (0, 94, 9),      # RGB for "#005e09"
        HexType.PASTURE: (82, 255, 98),  # RGB for "#52ff62"
        HexType.HILLS: (204, 31, 12),    # RGB for "#cc1f0c"
        HexType.MOUNTAINS: (122, 122, 122),  # RGB for "#7a7a7a"
        HexType.DESERT: (255, 229, 163),  # RGB for "#ffe5a3"
    }

    DEFAULT_RESOURCE_COLORS = {
        Resource.GRAIN: (255, 234, 41),  # RGB for "#ffea29"
        Resource.LUMBER: (0, 94, 9),      # RGB for "#005e09"
        Resource.WOOL: (82, 255, 98),  # RGB for "#52ff62"
        Resource.BRICK: (204, 31, 12),    # RGB for "#cc1f0c"
        Resource.ORE: (122, 122, 122),  # RGB for "#7a7a7a"
    }

    WATER_COLOR = (35, 135, 222)  # RGB for "#2387de"


    def __init__(
        self,
        board: _board.Board,
        player_color_map: Optional[Dict[Player, str]] = {},
        hex_color_map: Optional[Dict[HexType, str]] = DEFAULT_HEX_COLORS,
        resource_color_map: Optional[Dict[Resource, str]] = DEFAULT_RESOURCE_COLORS,
    ):
        self.board = board
        self._unused_player_colors = BoardRenderer.DEFAULT_PLAYER_COLORS[:]
        self.player_color_map = player_color_map
        self.hex_color_map = hex_color_map
        self.resource_color_map = resource_color_map

    def _get_player_color(self, player: Player):
        if player not in self.player_color_map:
            self.player_color_map[player] = self._unused_player_colors.pop(0)
        return self.player_color_map[player]

    def _get_path(self, chars, path, path_labels):
        fore = (156, 117, 0)
        back = self.hex_color_map[HexType.DESERT]
        if path.building is not None:
            fore = self._get_player_color(path.building.owner)
        elif frozenset(path.path_coords) in self.board.harbors:
            fore = (0, 0, 0)
        if path in path_labels:
            chars = [path_labels[path]] * len(chars)
        return list(map(lambda x: stylize(x, fore_rgb(fore[0], fore[1], fore[2]) + back_rgb(back[0], back[1], back[2])), chars))

    def _get_intersection(self, char, intersection, intersection_labels):
        fore = (156, 117, 0)
        back = self.hex_color_map[HexType.DESERT]
        if intersection in intersection_labels:
            return [
                stylize(intersection_labels[intersection], fore_rgb(0, 0, 0) + back_rgb(back[0], back[1], back[2]))
            ]
        if intersection.building is not None:
            fore = self._get_player_color(intersection.building.owner)
            char = (
                "s"
                if intersection.building.building_type is BuildingType.SETTLEMENT
                else "c"
            )
        return [stylize(char, fore_rgb(fore[0], fore[1], fore[2]) + back_rgb(back[0], back[1], back[2]))]

    def _get_hex_center(self, h, hex_labels):
        space = stylize(" ", back_rgb(self.hex_color_map[h.hex_type][0], self.hex_color_map[h.hex_type][1], self.hex_color_map[h.hex_type][2]))
        if h in hex_labels:
            return [space, space, hex_labels[h], space, space]
        if h.token_number is None:
            return [space] * 5
        token_color = (
            (255, 0, 0) if h.token_number == 6 or h.token_number == 8 else (0, 0, 0)
        )
        token_chars = [space if h.token_number < 10 else ""] + [
            stylize(h.token_number, fore_rgb(token_color[0], token_color[1], token_color[2]) + back_rgb(self.hex_color_map[h.hex_type][0], self.hex_color_map[h.hex_type][1], self.hex_color_map[h.hex_type][2]))
        ]
        return [space] + [t for t in token_chars] + [space, space]

    def _get_hex(self, coords, hex_labels, intersection_labels, path_labels):
        intersection_coords = [
            c + coords
            for c in (
                Coords(1, -1),
                Coords(1, 0),
                Coords(0, 1),
                Coords(-1, 1),
                Coords(-1, 0),
                Coords(0, -1),
            )
        ]
        intersections = [self.board.intersections[c] for c in intersection_coords]
        paths = [
            self.board.paths[
                frozenset(
                    {
                        intersection_coords[i],
                        intersection_coords[(i + 1) % len(intersection_coords)],
                    }
                )
            ]
            for i in range(len(intersection_coords))
        ]
        return [
            self._get_intersection(".", intersections[0], intersection_labels)
            + self._get_path(["-", "-"], paths[0], path_labels)
            + self._get_intersection("'", intersections[1], intersection_labels)
            + self._get_path(["-", "-"], paths[1], path_labels)
            + self._get_intersection(".", intersections[2], intersection_labels),
            self._get_path(["|"], paths[5], path_labels)
            + self._get_hex_center(self.board.hexes[coords], hex_labels)
            + self._get_path(["|"], paths[2], path_labels),
            self._get_intersection("'", intersections[5], intersection_labels)
            + self._get_path(["-", "-"], paths[4], path_labels)
            + self._get_intersection(".", intersections[4], intersection_labels)
            + self._get_path(["-", "-"], paths[3], path_labels)
            + self._get_intersection("'", intersections[3], intersection_labels),
        ]

    def _stylize_arr(self, arr, styles):
        return [stylize(s, styles) for s in arr]

    def _get_harbor(self, harbor):
        fore = (255, 255, 255) if harbor.resource is None else self.resource_color_map[harbor.resource]
        return [
            [stylize("3" if harbor.resource is None else "2", fore_rgb(fore[0], fore[1], fore[2]) + back_rgb(self.WATER_COLOR[0], self.WATER_COLOR[1], self.WATER_COLOR[2]))]
        ]

    def _get_harbor_coords(self, harbor):
        connected_coords = [
            [c + coord for c in Hex.CONNECTED_CORNER_OFFSETS]
            for coord in harbor.path_coords
        ]
        overlap = [
            c
            for c in connected_coords[0]
            if c in connected_coords[1] and c not in self.board.hexes
        ]
        hex_coords = self._get_hex_center_coords(overlap[0])
        return (hex_coords[0] + 2, hex_coords[1] + 1)

    def _copy_into_array(self, buf, to_copy, x, y):
        for i in range(len(to_copy)):
            for j in range(len(to_copy[i])):
                buf[y + i][x + j] = to_copy[i][j]

    def _get_hex_center_coords(self, coords):
        return ((int)(3 * coords.r), -(int)(1.34 * coords.q + 0.67 * coords.r))

    def get_coords_as_xy(self, coords: Coords) -> Tuple:
        """Get the coordinates given as x, y position.

        Args:
            coords: The coordinates
        Returns:
            The (x, y) position
        """
        if coords in self.board.hexes:
            x, y = self._get_hex_center_coords(coords)
            return (x + 2, y + 1)
        elif coords in self.board.intersections:
            h = list(self.board.get_hexes_connected_to_intersection(coords))[0]
            y, x = {
                Coords(1, 0): (3, 0),
                Coords(0, 1): (6, 0),
                Coords(-1, 1): (6, 2),
                Coords(-1, 0): (3, 2),
                Coords(0, -1): (0, 2),
                Coords(1, -1): (0, 0),
            }[coords - h]
            hy, hx = self._get_hex_center_coords(h)
            return (x + hx, y + hy)
        return 0, 0

    def get_board_as_string(
        self,
        hex_labels: Optional[Dict[Hex, str]] = {},
        intersection_labels: Optional[Dict[Intersection, str]] = {},
        path_labels: Optional[Dict[Path, str]] = {},
    ) -> str:
        """Get the board as a large, multiline string that includes colors.

        Args:
            hex_labels: A dictionary of labels to put on the hexes instead of the numbered tokens
            intersection_labels: A dictionary of labels to put on the points
            path_labels: A dictionary of labels to put on the paths

        Returns:
            str: The board as a string
        """
        size = 20, 55
        buf = [
            [stylize(" ", back_rgb(BoardRenderer.WATER_COLOR[0], BoardRenderer.WATER_COLOR[1], BoardRenderer.WATER_COLOR[2])) for j in range(size[1])]
            for i in range(size[0])
        ]

        center = int(size[1] / 2) - 3, int(size[0] / 2) - 1

        for hex_coords in self.board.hexes:
            x, y = self._get_hex_center_coords(hex_coords)
            self._copy_into_array(
                buf,
                self._get_hex(hex_coords, hex_labels, intersection_labels, path_labels),
                center[0] + x,
                center[1] + y,
            )
        for harbor in self.board.harbors.values():
            x, y = self._get_harbor_coords(harbor)
            self._copy_into_array(
                buf, self._get_harbor(harbor), center[0] + x, center[1] + y
            )

        x, y = self._get_hex_center_coords(self.board.robber)
        self._copy_into_array(
            buf,
            [[stylize("R", fore_rgb(255, 255, 255) + back_rgb(0, 0, 0))]],
            center[0] + x + 4,
            center[1] + y + 1,
        )

        return "\n".join(["".join(row) for row in buf])

    def render_board(
        self,
        hex_labels: Optional[Dict[Hex, str]] = {},
        intersection_labels: Optional[Dict[Intersection, str]] = {},
        path_labels: Optional[Dict[Path, str]] = {},
    ):
        """Render the board into the terminal.

        Args:
            hex_labels: A dictionary of labels to put on the hexes instead of the numbered tokens
            intersection_labels: A dictionary of labels to put on the points
            path_labels: A dictionary of labels to put on the paths
        """
        buf = self.get_board_as_string(hex_labels, intersection_labels, path_labels)
        print(buf)
