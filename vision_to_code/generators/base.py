from abc import ABC, abstractmethod
import textwrap
import numpy as np
import uuid

class BaseGenerator(ABC):
    def __init__(self, config):
        self.config = config
        self.resolution = config.resolution

    @abstractmethod
    def generate(self):
        """
        Returns a tuple: (code_str, image_array, metadata_dict)
        """
        pass

    def _prepare_block(self, text, indent_str="    "):
        """Indents a block of code, but strips indentation from the first line."""
        indented = textwrap.indent(textwrap.dedent(text).strip(), indent_str)
        return indented.lstrip()

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

    def _create_metadata(self, category, subtype, params):
        return {
            "id": str(uuid.uuid4()),
            "category": category,
            "subtype": subtype,
            "resolution": self.resolution,
            "params": params
        }
