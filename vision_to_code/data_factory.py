import numpy as np
import random
import textwrap

class DataFactory:
    def __init__(self, resolution=128):
        self.resolution = resolution

    def generate_code_and_image(self, category):
        if category == "2d":
            return self.generate_2d()
        elif category == "math":
            return self.generate_math()
        elif category == "fractal":
            return self.generate_fractal()
        elif category == "3d":
            return self.generate_3d()
        else:
            raise ValueError(f"Unknown category: {category}")

    def _exec_code(self, code_str):
        """Executes the generated code and extracts the 'img' variable."""
        context = {}
        try:
            exec(code_str, context)
        except Exception as e:
            print(f"Error executing generated code:\n{code_str}")
            raise e

        if 'img' not in context:
            raise ValueError("Generated code did not produce an 'img' variable.")

        img = context['img']
        return np.array(img, dtype=np.float32)

    def _prepare_block(self, text, indent_str="    "):
        """Indents a block of code, but strips indentation from the first line
        so it can be injected into a template that already has indentation."""
        indented = textwrap.indent(textwrap.dedent(text).strip(), indent_str)
        return indented.lstrip()

    def generate_2d(self):
        shape_type = random.choice(["circle", "square", "ellipse", "annulus", "triangle"])

        cx = random.uniform(-0.5, 0.5)
        cy = random.uniform(-0.5, 0.5)

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
            logic = f"mask = (x - ({cx:.2f}))**2 + (y - ({cy:.2f}))**2 < {r:.2f}**2"

        elif shape_type == "square":
            size = random.uniform(0.3, 0.8)
            logic = f"mask = np.maximum(np.abs(x - ({cx:.2f})), np.abs(y - ({cy:.2f}))) < {size:.2f}"

        elif shape_type == "ellipse":
            rx = random.uniform(0.3, 0.8)
            ry = random.uniform(0.3, 0.8)
            logic = f"mask = ((x - ({cx:.2f}))**2 / {rx:.2f}**2) + ((y - ({cy:.2f}))**2 / {ry:.2f}**2) < 1.0"

        elif shape_type == "annulus":
            r_outer = random.uniform(0.5, 0.9)
            r_inner = random.uniform(0.2, 0.4)
            if r_inner >= r_outer: r_inner = r_outer - 0.1
            logic = f"dist_sq = (x - ({cx:.2f}))**2 + (y - ({cy:.2f}))**2\n    mask = (dist_sq < {r_outer:.2f}**2) & (dist_sq > {r_inner:.2f}**2)"

        elif shape_type == "triangle":
            # Equilateral triangle pointing up
            # SDF: max(abs(q.x) * 0.866025 + q.y * 0.5, -q.y) - r * 0.5
            r = random.uniform(0.3, 0.6)
            logic = f"""
                qx = np.abs(x - ({cx:.2f}))
                qy = (y - ({cy:.2f}))
                d = np.maximum(qx * 0.866025 + qy * 0.5, -qy) - {r:.2f} * 0.5
                mask = d < 0
            """
            logic = self._prepare_block(logic, "    ")

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        return code, img

    def generate_math(self):
        func_type = random.choice(["parabola", "abs", "sine", "step"])

        code_template = textwrap.dedent(f"""
            import numpy as np

            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                # Normalize
                x = (x - res / 2) / (res / 4) # Zoom in a bit
                y = (y - res / 2) / (res / 4)

                {{logic}}

                # Create a line plot effect
                thickness = 0.1
                mask = np.abs(y - val) < thickness

                return mask.astype(float)

            img = render()
        """)

        if func_type == "parabola":
            a = random.uniform(0.5, 2.0)
            logic = f"val = {a:.2f} * x**2"
        elif func_type == "abs":
            logic = "val = np.abs(x)"
        elif func_type == "sine":
            freq = random.uniform(1.0, 3.0)
            logic = f"val = np.sin({freq:.2f} * x)"
        elif func_type == "step":
             logic = "val = np.sign(x)"

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        return code, img

    def generate_fractal(self):
        frac_type = random.choice(["mandelbrot", "julia"])

        # Parameters
        zoom = random.uniform(0.5, 1.5)
        if frac_type == "mandelbrot":
             cx, cy = -0.5, 0.0
             cx += random.uniform(-0.2, 0.2)
             cy += random.uniform(-0.2, 0.2)
        else:
             cx, cy = 0.0, 0.0

        max_iter = random.randint(20, 50)

        # We use {{setup_z_c}} so f-string doesn't replace it yet.
        # Dedent happens first, then we replace.
        code_template = textwrap.dedent(f"""
            import numpy as np

            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                # Map to complex plane
                # Scale: 3.0 / zoom
                scale = 3.0 / {zoom:.2f}
                x = (x - res / 2) / res * scale + {cx:.2f}
                y = (y - res / 2) / res * scale + {cy:.2f}

                c = x + 1j * y
                z = np.zeros_like(c)

                {{setup_z_c}}

                img = np.zeros(c.shape)

                for i in range({max_iter}):
                    mask = np.abs(z) < 10
                    z[mask] = z[mask]**2 + c[mask]
                    img[mask] += 1

                return img / {max_iter}.0

            img = render()
        """)

        if frac_type == "mandelbrot":
            setup = "pass # c is coordinate, z starts at 0"
        else: # Julia
            jx = random.uniform(-1, 1)
            jy = random.uniform(-1, 1)
            setup = f"""
            # Julia set constant
            julia_c = {jx:.3f} + {jy:.3f}j
            z = c # Start z at coordinate
            c = np.full_like(z, julia_c) # c is constant
            """

        # Helper prepares the block to match indentation of 16 spaces (inside def render)
        # Wait, inside def render is 4 spaces.
        setup = self._prepare_block(setup, "    ")

        code = code_template.replace("{setup_z_c}", setup)
        img = self._exec_code(code)
        return code, img

    def generate_3d(self):
        shape_type = random.choice(["sphere", "cube"])

        code_template = textwrap.dedent(f"""
            import numpy as np

            def render(res={self.resolution}):
                # Normalized Device Coordinates (NDC)
                y, x = np.ogrid[:res, :res]
                uv_x = (x - res / 2) / (res / 2)
                uv_y = (y - res / 2) / (res / 2)

                # Broadcast to grid
                uv_x, uv_y = np.broadcast_arrays(uv_x, uv_y)

                # Ray Origin and Direction
                ro = np.array([0.0, 0.0, -3.0])
                # Simple perspective: screen is at z=-1 approx
                rd = np.stack((uv_x, uv_y, np.ones_like(uv_x)), axis=-1)
                # Normalize rd
                norm = np.linalg.norm(rd, axis=-1, keepdims=True)
                rd = rd / norm

                # Image buffer (depth or mask)
                img = np.zeros((res, res))

                # Raymarching Loop
                t = np.zeros((res, res))
                hits = np.zeros((res, res), dtype=bool)

                for i in range(30):
                    p = ro + rd * t[..., np.newaxis]

                    # SDF Evaluation
                    {{sdf_logic}}

                    # Step
                    t += d

                    # Check hit or miss
                    # If d is very small, we hit
                    hit_mask = d < 0.01
                    hits |= hit_mask

                    # If d is too large, we missed

                # Simple lighting based on t (depth) or just silhouette
                # Let's do silhouette + some depth gradient
                final = hits.astype(float) * (1.0 - (t - 2.0) / 3.0)
                final = np.clip(final, 0, 1)
                return final

            img = render()
        """)

        if shape_type == "sphere":
            r = random.uniform(0.8, 1.2)
            sdf_logic = f"""
            # Sphere SDF
            d = np.linalg.norm(p, axis=-1) - {r:.2f}
            """
        elif shape_type == "cube":
            s = random.uniform(0.6, 0.9)
            sdf_logic = f"""
            # Cube SDF
            # Axis aligned cube
            q = np.abs(p) - {s:.2f}
            d = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[...,0], np.maximum(q[...,1], q[...,2])), 0.0)
            """

        # Indent 8 spaces for inside the loop
        sdf_logic = self._prepare_block(sdf_logic, "        ")

        code = code_template.replace("{sdf_logic}", sdf_logic)
        img = self._exec_code(code)
        return code, img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    factory = DataFactory()
    os.makedirs("output_samples", exist_ok=True)

    # Generate one of each category
    categories = ["2d", "math", "fractal", "3d"]
    for cat in categories:
        print(f"Generating {cat}...")
        try:
            code, img = factory.generate_code_and_image(cat)

            plt.imsave(f"output_samples/{cat}_sample.png", img, cmap='gray')
            with open(f"output_samples/{cat}_code.py", "w") as f:
                f.write(code)
        except Exception as e:
            print(f"Failed to generate {cat}: {e}")

    print("Samples generated in output_samples/")
