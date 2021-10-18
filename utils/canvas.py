"""
Utilities relating to rendering SVG images from offset data
"""

import numpy as np
import svgwrite as svg

from utils.data import strokes_to_lines


class CanvasGrid:
    def __init__(self, filename, cell_size=255, nrows=1, ncols=1, padding=(50, 50)):
        self.cell_size = cell_size
        self.padding = padding
        self.nrows = nrows
        self.ncols = ncols
        self.drw = svg.Drawing(filename, (cell_size * nrows, cell_size * ncols))

        # Background color
        self.drw.add(
            self.drw.rect(
                insert=(0, 0), size=("100%", "100%"), rx=None, ry=None, fill="white"
            )
        )

    def _cell_offset(self, row, col):
        if row > self.nrows:
            raise IndexError(f"Row index {row} out of bounds")
        if col > self.ncols:
            raise IndexError(f"Column index {col} out of bounds")

        return (self.cell_size * col, self.cell_size * row)

    def _normalize_lines(self, lines):
        # Find the bounding box of the canvas for normalizing
        xs = [x for line in lines for x, _ in line]
        ys = [y for line in lines for _, y in line]

        min_x, max_x = (min(xs), max(xs))
        min_y, max_y = (min(ys), max(ys))

        def _normalize(x, y):
            res_x = self.padding[0] + (self.cell_size - 2 * self.padding[0]) * (
                    x - min_x
            ) / (max_x - min_x)
            res_y = self.padding[1] + (self.cell_size - 2 * self.padding[1]) * (
                    y - min_y
            ) / (max_y - min_y)
            return [res_x, res_y]

        return [[_normalize(x, y) for x, y in line] for line in lines]

    def _render_lines(self, lines, row, col, color):
        (dy, dx) = self._cell_offset(row, col)

        for line in lines:
            for i in range(1, len(line)):
                x0, y0 = line[i-1]
                x1, y1 = line[i]
                p0 = (x0 + dx, y0 + dy)
                p1 = (x1 + dx, y1 + dy)
                self.drw.add(self.drw.line(p0, p1, stroke=color))

    def draw_strokes(self, strokes, row, col, color="black"):
        lines = strokes_to_lines(strokes)
        normalized = self._normalize_lines(lines)
        self._render_lines(normalized, row, col, color)

    def save(self):
        self.drw.save()
