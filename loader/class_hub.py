"""
class_hub.py

Dynamic registry / plugin loader for the project.

The idea is simple:  every “component family” (Operators, Predictors,
Processors, Embedders …) lives inside its own package, follows a common
naming convention *and* inherits from a shared base-class.

Given that structure the `ClassHub` utility
    1) scans the corresponding directory for Python files that match
       the pattern `*_{type}.py`
    2) imports those modules dynamically,
    3) collects every concrete subclass of the declared `base_class`,
    4) exposes them through a dict-like interface that is keyed by their
       **lower-case name without the type suffix**  
       (e.g.  “MLPEmbedder” → "mlp").

Example
-------
>>> embedder_cls = ClassHub.embedders()["optlarge"]
>>> embedder     = embedder_cls()

Design notes
------------
• The public factory methods (`operators()`, `predictors()` …) are mere
  convenience wrappers that pre-configure the hub for a specific
  component family.

• `__call__`, `__getitem__` and `__contains__` are implemented so that
  an instance behaves similarly to a dictionary.
"""

from __future__ import annotations

import glob
import importlib
import os.path
from pathlib import Path
from typing import Dict, List, Type


class ClassHub:
    # ----------------------------- #
    # Convenience factory methods   #
    # ----------------------------- #
    @staticmethod
    def operators() -> "ClassHub":
        """
        Registry for every subclass of `model.operators.base_operator.BaseOperator`
        found in the *model/operators* package.
        """
        from model.operators.base_operator import BaseOperator  # local import to avoid circular deps
        return ClassHub(BaseOperator, os.path.join("model", "operators"), "Operator")

    @staticmethod
    def predictors() -> "ClassHub":
        """
        Registry for every subclass of `model.predictors.base_predictor.BasePredictor`
        found in the *model/predictors* package.
        """
        from model.predictors.base_predictor import BasePredictor
        return ClassHub(BasePredictor, os.path.join("model", "predictors"), "Predictor")

    @staticmethod
    def processors() -> "ClassHub":
        """
        Registry for `processor` package – subclasses of
        `processor.base_processor.BaseProcessor`.
        """
        from processor.base_processor import BaseProcessor
        return ClassHub(BaseProcessor, "processor", "Processor")

    @staticmethod
    def embedders() -> "ClassHub":
        """
        Registry for `embedder` package – subclasses of
        `embedder.base_embedder.BaseEmbedder`.
        """
        from embedder.base_embedder import BaseEmbedder
        return ClassHub(BaseEmbedder, "embedder", "Embedder")

    # ----------------------------- #
    # Construction                  #
    # ----------------------------- #
    def __init__(self,
                 base_class: Type,
                 module_dir: str,
                 module_type: str) -> None:
        """
        Parameters
        ----------
        base_class : Type
            The common parent class that every loadable component must inherit from.
            (e.g. `BaseOperator`, `BasePredictor`, …)
        module_dir : str
            Package / directory that contains the concrete implementations.
            May use “/” on any platform; will be normalised via `Path`.
        module_type : str
            Readable suffix used in the file- and class-name convention,
            e.g.  "Operator"  → files like *xyz_operator.py* and classes like *XYZOperator*.
        """
        self.base_class = base_class
        self.module_dir = module_dir
        self.module_type = module_type.lower()      # e.g. "operator"
        # Capitalised form (first letter upper, rest identical) → "Operator"
        self.upper_module_type = self.module_type[0].upper() + self.module_type[1:]

        # Scan the directory and build both list and dict representations
        self.class_list: List[Type] = self._get_class_list()
        self.class_dict: Dict[str, Type] = {}
        for cls in self.class_list:
            # Drop the suffix (e.g. *XYZOperator* → "XYZ") and lower-case the key
            key = cls.__name__.replace(self.upper_module_type, "").lower()
            self.class_dict[key] = cls

    # ----------------------------- #
    # Implementation details        #
    # ----------------------------- #
    def _get_class_list(self) -> List[Type]:
        """
        Discover concrete subclasses in `self.module_dir` that match the
        *file-name* pattern '*_{module_type}.py'.

        Returns
        -------
        list[type]
            All discovered subclasses (order corresponds to `glob` results).
        """
        # Platform-independent glob: path/to/dir/*_operator.py
        file_pattern = str(Path(self.module_dir) / f"*_{self.module_type}.py")
        class_list: List[Type] = []

        for file_path in glob.glob(file_pattern):
            # Convert file path into importable module path
            path_obj = Path(file_path)
            file_name = path_obj.stem  # "xyz_operator"
            # `parts` gives ('model', 'operators') -> join with '.' and add file name
            module_path = ".".join(Path(self.module_dir).parts + (file_name,))

            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Collect suitable classes defined in that module
            for name, obj in module.__dict__.items():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, self.base_class)
                    and obj is not self.base_class
                ):
                    class_list.append(obj)

        return class_list

    # ----------------------------- #
    # Dict-like public interface    #
    # ----------------------------- #
    def __call__(self, name: str):
        """
        Allow direct call syntax:  hub("mlp") ⇢ <class 'MLPEmbedder'>
        """
        return self.class_dict[name.lower()]

    # Item access  hub["mlp"]
    __getitem__ = __call__

    def __contains__(self, name: str) -> bool:
        """
        `name in hub`  → boolean
        """
        return name.lower() in self.class_dict

    def list(self):
        """
        Convenience helper to list all registered keys.
        """
        return list(self.class_dict.keys())
