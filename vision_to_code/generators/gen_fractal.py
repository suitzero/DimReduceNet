import random
import textwrap
import numpy as np
from .base import BaseGenerator

class GenFractal(BaseGenerator):
    def generate(self):
        frac_type = random.choice(["mandelbrot", "julia"])

        zoom = random.uniform(0.5, 1.5)
        if frac_type == "mandelbrot":
             cx, cy = -0.5, 0.0
             cx += random.uniform(-0.2, 0.2)
             cy += random.uniform(-0.2, 0.2)
        else:
             cx, cy = 0.0, 0.0

        max_iter = random.randint(20, 50)

        params = {
            "zoom": zoom,
            "cx": cx,
            "cy": cy,
            "max_iter": max_iter,
            "type": frac_type
        }

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
            params["julia_constant"] = {"real": jx, "imag": jy}
            setup = f"""
            # Julia set constant
            julia_c = {jx:.3f} + {jy:.3f}j
            z = c # Start z at coordinate
            c = np.full_like(z, julia_c) # c is constant
            """

        setup = self._prepare_block(setup, "    ")

        code = code_template.replace("{setup_z_c}", setup)
        img = self._exec_code(code)
        metadata = self._create_metadata("fractal", frac_type, params)
        return code, img, metadata
