import random
import textwrap
import numpy as np
from .base import BaseGenerator

class GenMath(BaseGenerator):
    def generate(self):
        func_type = random.choice(["parabola", "abs", "sine", "step"])
        params = {"func": func_type}

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
            params["a"] = a
            logic = f"val = {a:.2f} * x**2"
        elif func_type == "abs":
            logic = "val = np.abs(x)"
        elif func_type == "sine":
            freq = random.uniform(1.0, 3.0)
            params["freq"] = freq
            logic = f"val = np.sin({freq:.2f} * x)"
        elif func_type == "step":
             logic = "val = np.sign(x)"

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        metadata = self._create_metadata("math", func_type, params)

        return code, img, metadata
