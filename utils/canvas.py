"""
Utilities relating to rendering SVG images from offset data
"""

import numpy as np
import svgwrite as svg
import matplotlib.pyplot as plt
import PIL

from utils.data import strokes_to_lines


class CanvasGrid:
    def __init__(self, filename, cell_size=255, nrows=1, ncols=1, padding=(50, 50)):
        self.cell_size = cell_size
        self.padding = padding
        self.nrows = nrows
        self.ncols = ncols
        self.drw = svg.Drawing(filename, (cell_size * ncols, cell_size * nrows))

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
        (dx, dy) = self._cell_offset(row, col)

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


def strokes_to_rgb(S):
    # Code adapted from https://github.com/OhataKenji/SketchRNN-Pytorch/tree/7b7dd319df0ae0a521e4eb2a9c1733b3d0a535d1
    plt.axis('equal')
    lines = strokes_to_lines(S)

    for line in lines:
        for i in range(1, len(line)):
            x0, y0 = line[i-1]
            x1, y1 = line[i]
            plt.plot([x0, x1], [-y0, -y1])

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    plt.close("all")

    return np.asarray(pil_image)
