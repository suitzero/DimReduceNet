import numpy as np
import random
import textwrap

class SyntheticDataFactory:
    def __init__(self, resolution=64):
        self.resolution = resolution

    def _exec_code(self, code_str):
        context = {}
        try:
            exec(code_str, context)
        except Exception as e:
            print(f"Error executing generated code:\n{code_str}")
            raise e
        return np.array(context['img'], dtype=np.float32)

    def generate_2d_primitive(self):
        shape_type = random.choice(["circle", "square"])
        cx = random.uniform(-0.5, 0.5)
        cy = random.uniform(-0.5, 0.5)
        params = {"shape": shape_type, "cx": cx, "cy": cy}

        code_template = textwrap.dedent(f"""
            import numpy as np
            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
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

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        return params, code, img

    def generate_math_function(self):
        func_type = random.choice(["parabola", "sine"])
        params = {"func": func_type}

        code_template = textwrap.dedent(f"""
            import numpy as np
            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                x = (x - res / 2) / (res / 4)
                y = (y - res / 2) / (res / 4)
                {{logic}}
                thickness = 0.1
                mask = np.abs(y - val) < thickness
                return mask.astype(float)
            img = render()
        """)

        if func_type == "parabola":
            a = random.uniform(0.5, 2.0)
            params["a"] = a
            logic = f"val = {a:.2f} * x**2"
        elif func_type == "sine":
            freq = random.uniform(1.0, 3.0)
            params["freq"] = freq
            logic = f"val = np.sin({freq:.2f} * x)"

        code = code_template.replace("{logic}", logic)
        img = self._exec_code(code)
        return params, code, img

    def generate_3d_primitive(self):
        shape_type = random.choice(["sphere", "cube"])
        params = {"shape": shape_type}

        code_template = textwrap.dedent(f"""
            import numpy as np
            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                uv_x = (x - res / 2) / (res / 2)
                uv_y = (y - res / 2) / (res / 2)
                uv_x, uv_y = np.broadcast_arrays(uv_x, uv_y)
                ro = np.array([0.0, 0.0, -3.0])
                rd = np.stack((uv_x, uv_y, np.ones_like(uv_x)), axis=-1)
                norm = np.linalg.norm(rd, axis=-1, keepdims=True)
                rd = rd / norm
                t = np.zeros((res, res))
                hits = np.zeros((res, res), dtype=bool)
                for i in range(30):
                    p = ro + rd * t[..., np.newaxis]
                    {{sdf_logic}}
                    t += d
                    hits |= (d < 0.01)
                final = hits.astype(float) * (1.0 - (t - 2.0) / 3.0)
                return np.clip(final, 0, 1)
            img = render()
        """)

        if shape_type == "sphere":
            r = random.uniform(0.8, 1.2)
            params["r"] = r
            sdf_logic = f"d = np.linalg.norm(p, axis=-1) - {r:.2f}"
        elif shape_type == "cube":
            s = random.uniform(0.6, 0.9)
            params["s"] = s
            sdf_logic = f"q = np.abs(p) - {s:.2f}\n        d = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[...,0], np.maximum(q[...,1], q[...,2])), 0.0)"

        code = code_template.replace("{sdf_logic}", sdf_logic)
        img = self._exec_code(code)
        return params, code, img

    def generate_composite_csg_scene(self):
        shape1 = random.choice(["sphere", "cube"])
        shape2 = random.choice(["sphere", "cube"])
        op = random.choice(["union", "intersection", "difference"])
        
        params = {"shape1": shape1, "shape2": shape2, "op": op}
        
        code_template = textwrap.dedent(f"""
            import numpy as np
            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                uv_x = (x - res / 2) / (res / 2)
                uv_y = (y - res / 2) / (res / 2)
                uv_x, uv_y = np.broadcast_arrays(uv_x, uv_y)
                ro = np.array([0.0, 0.0, -3.0])
                rd = np.stack((uv_x, uv_y, np.ones_like(uv_x)), axis=-1)
                norm = np.linalg.norm(rd, axis=-1, keepdims=True)
                rd = rd / norm
                t = np.zeros((res, res))
                hits = np.zeros((res, res), dtype=bool)
                for i in range(30):
                    p = ro + rd * t[..., np.newaxis]
                    {{sdf_logic}}
                    t += d
                    hits |= (d < 0.01)
                final = hits.astype(float) * (1.0 - (t - 2.0) / 3.0)
                return np.clip(final, 0, 1)
            img = render()
        """)
        
        def get_sdf_logic(shape_type, var_prefix, idx):
            if shape_type == "sphere":
                r = random.uniform(0.6, 1.0)
                params[f"r{idx}"] = r
                return f"{var_prefix} = np.linalg.norm(p, axis=-1) - {r:.2f}"
            elif shape_type == "cube":
                s = random.uniform(0.5, 0.8)
                params[f"s{idx}"] = s
                return f"q{idx} = np.abs(p) - {s:.2f}\n        {var_prefix} = np.linalg.norm(np.maximum(q{idx}, 0.0), axis=-1) + np.minimum(np.maximum(q{idx}[...,0], np.maximum(q{idx}[...,1], q{idx}[...,2])), 0.0)"

        sdf1 = get_sdf_logic(shape1, "d1", 1)
        sdf2 = get_sdf_logic(shape2, "d2", 2)
        
        if op == "union":
            combine_logic = "d = np.minimum(d1, d2)"
        elif op == "intersection":
            combine_logic = "d = np.maximum(d1, d2)"
        elif op == "difference":
            combine_logic = "d = np.maximum(d1, -d2)"
            
        sdf_logic = f"{sdf1}\n        {sdf2}\n        {combine_logic}"
        
        code = code_template.replace("{sdf_logic}", sdf_logic)
        img = self._exec_code(code)
        return params, code, img

    
    def generate_multi_object_scene(self, min_objects=2, max_objects=3):
        num_objects = random.randint(min_objects, max_objects)
        params = {"num_objects": num_objects}
        
        code_template = textwrap.dedent(f"""
            import numpy as np
            def render(res={self.resolution}):
                y, x = np.ogrid[:res, :res]
                uv_x = (x - res / 2) / (res / 2)
                uv_y = (y - res / 2) / (res / 2)
                uv_x, uv_y = np.broadcast_arrays(uv_x, uv_y)
                ro = np.array([0.0, 0.0, -3.0])
                rd = np.stack((uv_x, uv_y, np.ones_like(uv_x)), axis=-1)
                norm = np.linalg.norm(rd, axis=-1, keepdims=True)
                rd = rd / norm
                t = np.zeros((res, res))
                hits = np.zeros((res, res), dtype=bool)
                for i in range(30):
                    p = ro + rd * t[..., np.newaxis]
                    {{sdf_logic}}
                    t += d
                    hits |= (d < 0.01)
                final = hits.astype(float) * (1.0 - (t - 2.0) / 3.0)
                return np.clip(final, 0, 1)
            img = render()
        """)
        
        def get_sdf_logic(idx):
            shape_type = random.choice(["sphere", "cube", "cylinder"])
            cx = random.uniform(-1.5, 1.5)
            params[f"shape{idx}"] = shape_type
            params[f"cx{idx}"] = cx
            
            p_shifted = f"p_shifted_{idx} = p - np.array([{cx:.2f}, 0.0, 0.0])"
            
            if shape_type == "sphere":
                r = random.uniform(0.4, 0.8)
                params[f"r{idx}"] = r
                sdf = f"d{idx} = np.linalg.norm(p_shifted_{idx}, axis=-1) - {r:.2f}"
            elif shape_type == "cube":
                s = random.uniform(0.3, 0.6)
                params[f"s{idx}"] = s
                sdf = f"q{idx} = np.abs(p_shifted_{idx}) - {s:.2f}\n        d{idx} = np.linalg.norm(np.maximum(q{idx}, 0.0), axis=-1) + np.minimum(np.maximum(q{idx}[...,0], np.maximum(q{idx}[...,1], q{idx}[...,2])), 0.0)"
            elif shape_type == "cylinder":
                r = random.uniform(0.3, 0.5)
                h = random.uniform(0.4, 0.8)
                params[f"r{idx}"] = r
                params[f"h{idx}"] = h
                sdf = f"d{idx} = np.maximum(np.linalg.norm(p_shifted_{idx}[..., [0,2]], axis=-1) - {r:.2f}, np.abs(p_shifted_{idx}[..., 1]) - {h:.2f})"
            return f"{p_shifted}\n        {sdf}"

        sdfs = []
        for i in range(num_objects):
            sdfs.append(get_sdf_logic(i+1))
            
        sdf_logic = "\n        ".join(sdfs)
        
        if num_objects == 2:
            combine_logic = "d = np.minimum(d1, d2)"
        else:
            combine_logic = "d = np.minimum(d1, np.minimum(d2, d3))"
            
        sdf_logic = f"{sdf_logic}\n        {combine_logic}"
        
        code = code_template.replace("{sdf_logic}", sdf_logic)
        img = self._exec_code(code)
        return params, code, img

    def sample_time_series(self, fn, xs, ts, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        X, T = np.meshgrid(xs, ts, indexing='ij')
        X_flat = X.flatten()
        T_flat = T.flatten()
        
        Y_flat = fn(X_flat, T_flat)
        return X_flat, T_flat, Y_flat

    def generate_bouncing_ball_trajectory(self, t_min=0.0, t_max=5.0, num_steps=50, A=1.5, f=1.0/(2*np.pi), phi=0.0):
        ts = np.linspace(t_min, t_max, num_steps)
        xs = np.zeros_like(ts)
        
        ys = A * np.abs(np.sin(2 * np.pi * f * ts + phi))
        
        params = {"A": A, "f": f, "phi": phi}
        
        return params, xs, ts, ys
