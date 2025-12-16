import random
import textwrap
import numpy as np
from .base import BaseGenerator

class Gen2D(BaseGenerator):
    def generate(self):
        shape_type = random.choice(["circle", "square", "ellipse", "annulus", "triangle", "pentagon", "star"])

        cx = random.uniform(-0.5, 0.5)
        cy = random.uniform(-0.5, 0.5)
        params = {"cx": cx, "cy": cy, "shape": shape_type}

        code_template = textwrap.dedent(f"""
            import numpy as np

            def render(res={self.resolution}):
                # Coordinate system
                y, x = np.ogrid[:res, :res]
                # Normalize coordinates to range [-1.5, 1.5]
                x = (x - res / 2) / (res / 2) * 1.5
                y = (y - res / 2) / (res / 2) * 1.5

                {{logic}}

                return mask.astype(float)

            img = render()
        """)

        if shape_type == "circle":
            r = random.uniform(0.3, 0.8)
            params["r"] = r
            logic = f"mask = (x - ({cx:.2f}))**2 + (y - ({cy:.2f}))**2 < {r:.2f}**2"

        elif shape_type == "square":
            size = random.uniform(0.3, 0.8)
            params["size"] = size
            logic = f"mask = np.maximum(np.abs(x - ({cx:.2f})), np.abs(y - ({cy:.2f}))) < {size:.2f}"

        elif shape_type == "ellipse":
            rx = random.uniform(0.3, 0.8)
            ry = random.uniform(0.3, 0.8)
            params["rx"] = rx
            params["ry"] = ry
            logic = f"mask = ((x - ({cx:.2f}))**2 / {rx:.2f}**2) + ((y - ({cy:.2f}))**2 / {ry:.2f}**2) < 1.0"

        elif shape_type == "annulus":
            r_outer = random.uniform(0.5, 0.9)
            r_inner = random.uniform(0.2, 0.4)
            if r_inner >= r_outer: r_inner = r_outer - 0.1
            params["r_outer"] = r_outer
            params["r_inner"] = r_inner
            logic = f"dist_sq = (x - ({cx:.2f}))**2 + (y - ({cy:.2f}))**2\n    mask = (dist_sq < {r_outer:.2f}**2) & (dist_sq > {r_inner:.2f}**2)"

        elif shape_type == "triangle":
            r = random.uniform(0.3, 0.6)
            params["r"] = r
            logic = f"""
                qx = np.abs(x - ({cx:.2f}))
                qy = (y - ({cy:.2f}))
                d = np.maximum(qx * 0.866025 + qy * 0.5, -qy) - {r:.2f} * 0.5
                mask = d < 0
            """
            logic = self._prepare_block(logic, "    ")

        elif shape_type == "pentagon":
            r = random.uniform(0.3, 0.6)
            params["r"] = r
            # Simple approximation using angle
            # SDF for regular polygon with N sides:
            # d = cos(floor(0.5 + a/slice)*slice - a) * r - size
            logic = f"""
                # Pentagon
                px = x - ({cx:.2f})
                py = y - ({cy:.2f})
                angle = np.arctan2(py, px) + np.pi
                r_pixel = np.sqrt(px**2 + py**2)

                # 5 sides
                slice = np.pi * 2.0 / 5.0
                angle_mod = np.mod(angle + slice * 0.5, slice) - slice * 0.5

                dist = np.cos(angle_mod) * r_pixel - {r:.2f} * 0.5
                mask = dist < 0
            """
            logic = self._prepare_block(logic, "    ")

        elif shape_type == "star":
            r = random.uniform(0.3, 0.6)
            params["r"] = r
            # Star
            logic = f"""
                # Star shape
                px = x - ({cx:.2f})
                py = y - ({cy:.2f})
                angle = np.arctan2(py, px)
                r_pixel = np.sqrt(px**2 + py**2)

                # Modulate radius with sine of angle
                # 5 points
                r_boundary = {r:.2f} * (0.6 + 0.4 * np.cos(5.0 * angle))
                mask = r_pixel < r_boundary
            """
            logic = self._prepare_block(logic, "    ")

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        metadata = self._create_metadata("2d", shape_type, params)

        return code, img, metadata
