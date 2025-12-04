import importlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import numpy as np
import pyarrow as pa

__all__ = ["Data"]

T = TypeVar("T", bound="Data")


class Data(ABC, Generic[T]):
    """
    Base class for all data objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.properties: Dict[str, Any] = {}

    def add_property(self, name: str, values: Any) -> None:
        """
        Add a property to the data object.

        Args:
            name (str): Name of the property.
            prop (Property): Property object.
        """
        if len(values) != len(self):
            raise ValueError(
                "Property must have the same length as the data object: "
                + f"{len(values)} != {len(self)}"
            )
        self.properties[name] = values

    def clear_properties(self):
        self.properties = {}

    def clear_property(self, name: str):
        try:
            del self.properties[name]
        except KeyError:
            raise ValueError(f"Property {name} does not exist.")

    @abstractmethod
    def __getitem__(self, idx: List[int] | int | np.ndarray) -> T:
        pass

    @abstractmethod
    def __setitem__(self, idx: List[int] | int | np.ndarray, value: T) -> None:
        pass

    def split(self, num_partitions: int) -> List[T]:
        """
        Split the data object into multiple partitions.

        Args:
            num_partitions (int): Number of partitions to split the data object into.

        Returns:
            List[T]: List of data objects.
        """
        idx = np.arange(len(self))
        indices = np.array_split(idx, num_partitions)
        # filter empty partitions
        indices = [idx for idx in indices if len(idx) > 0]
        data = [self[idx] for idx in indices]
        return data

    @staticmethod
    @abstractmethod
    def concatenate(data: List[T]) -> T:
        """
        Concatenate two data objects.

        Args:
            data (T): Data objects to concatenate.

        Returns:
            T: Concatenated data.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def ids(self) -> List[str]:
        """
        The columns of the arrow table that uniquely identify an entry.
        """
        pass

    def _add_properties_to_table(self, table: pa.Table) -> pa.Table:
        """
        Add properties to the arrow table.
        """
        for name, values in self.properties.items():
            prefix = "_prop_"
            if type(values[0]) in [dict, list]:
                values = [json.dumps(v) for v in values]
                prefix += "json_"
            elif type(values) in [np.ndarray]:
                prefix += "np_"
            arr = pa.array(values)
            table = table.append_column(prefix + name, arr)
        return table

    def _load_properties_from_table(self, table: pa.Table) -> None:
        """
        Parse properties from the arrow table.
        """
        for name in table.column_names:
            if name.startswith("_prop_"):
                if name.startswith("_prop_json_"):
                    self.properties[name[11:]] = [
                        json.loads(v) for v in table[name].to_pylist()
                    ]
                elif name.startswith("_prop_np_"):
                    self.properties[name[9:]] = table[name].to_numpy()
                else:
                    self.properties[name[6:]] = table[name].to_pylist()

    @abstractmethod
    def to_arrow(self) -> pa.Table:
        """
        Convert the data object to an arrow table.
        """
        pass

    @staticmethod
    def from_arrow(table: pa.Table) -> T:
        """
        Convert an arrow table to a data object.
        """
        class_path = table.schema.metadata[b"_decoder"]
        mpath = class_path.decode("utf8").split(".")
        module = importlib.import_module(".".join(mpath[:-1]))
        class_ = getattr(module, mpath[-1])
        data: T = class_.from_arrow(table)
        return data
