"""
Utilities relating to rendering SVG images from offset data
"""

import numpy as np
import svgwrite as svg


class CanvasGrid:
    def __init__(self, filename, cell_size=255, nrows=1, ncols=1):
        self.cell_size = cell_size
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

    def _normalize_strokes(self, strokes, padding=(50, 50)):
        x_min = np.array(strokes)[..., 0].min()
        x_max = np.array(strokes)[..., 0].max()
        y_min = np.array(strokes)[..., 1].min()
        y_max = np.array(strokes)[..., 1].max()

        normalized_strokes = np.zeros_like(strokes)

        # Rescale X
        normalized_strokes[..., 0] = padding[0] + (self.cell_size - 2 * padding[0]) * (
            strokes[..., 0] - x_min
        ) / (x_max - x_min)
        # Rescale Y
        normalized_strokes[..., 1] = padding[1] + (self.cell_size - 2 * padding[1]) * (
            strokes[..., 1] - y_min
        ) / (y_max - y_min)

        return normalized_strokes

    def _render_strokes(self, strokes, row, col, color):
        (dy, dx) = self._cell_offset(row, col)

        for (xs, ys) in strokes.tolist():
            p0 = (xs[0] + dx, ys[0] + dy)
            p1 = (xs[1] + dx, ys[1] + dy)
            self.drw.add(self.drw.line(p0, p1, stroke=color))

    def add_drawing(self, drawing_data, row, col, color="black"):
        x = np.cumsum(
            drawing_data, axis=0
        )  # Transforms offsets into coordinates, still need to append/start at (0,0)
        x = np.insert(x, 0, [0, 0, 0], axis=0)  # Should start in the origin
        sections = np.unique(x[..., 2], return_index=True)[
            1
        ]  # Only care about the indices (second in tuple)
        strokes = self._normalize_strokes(
            x[..., :2]
        )  # Normalize to fit within the output image
        stroke_sections = np.split(strokes, sections, axis=0)[
            1:
        ]  # First element is always empty, discard this

        # Render the strokes in each section
        for section_strokes in stroke_sections:
            self._render_strokes(
                np.dstack([section_strokes[:-1], section_strokes[1:]]), row, col, color
            )

    def save(self):
        self.drw.save()
