import random
import textwrap
import numpy as np
from .base import BaseGenerator

class Gen3D(BaseGenerator):
    def generate(self):
        shape_type = random.choice(["sphere", "cube", "ellipsoid"])
        params = {"shape": shape_type}

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
            params["r"] = r
            sdf_logic = f"""
            # Sphere SDF
            d = np.linalg.norm(p, axis=-1) - {r:.2f}
            """
        elif shape_type == "cube":
            s = random.uniform(0.6, 0.9)
            params["s"] = s
            sdf_logic = f"""
            # Cube SDF
            # Axis aligned cube
            q = np.abs(p) - {s:.2f}
            d = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[...,0], np.maximum(q[...,1], q[...,2])), 0.0)
            """
        elif shape_type == "ellipsoid":
            r1 = random.uniform(0.5, 0.8)
            r2 = random.uniform(0.5, 0.8)
            r3 = random.uniform(0.5, 0.8)
            params["r"] = [r1, r2, r3]
            # Simple scaling approximation for Ellipsoid
            # Not exact distance, but good enough for visualization
            # d = (length(p/r) - 1.0) * min(r)
            sdf_logic = f"""
            # Ellipsoid SDF (Approx)
            radii = np.array([{r1:.2f}, {r2:.2f}, {r3:.2f}])
            # Avoid division by zero, radii are > 0.5
            d = (np.linalg.norm(p / radii, axis=-1) - 1.0) * {min(r1, r2, r3):.2f}
            """

        sdf_logic = self._prepare_block(sdf_logic, "        ")

        code = code_template.replace("{sdf_logic}", sdf_logic)
        img = self._exec_code(code)
        metadata = self._create_metadata("3d", shape_type, params)
        return code, img, metadata
