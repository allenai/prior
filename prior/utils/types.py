import json
from typing import Any, List, Optional

# Literal was introduced in Python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from attr import field
from attrs import define


@define
class Dataset:
    data: List[Any]
    """The entries in the dataset."""

    dataset: str
    """The split of the dataset."""

    split: Literal["train", "val", "test"]
    """The split of the dataset."""

    def __iter__(self):
        """Return an iterator over the dataset."""
        for item in self.data:
            yield item

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        return self.data[index]

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (
            "Dataset(\n"
            f"    dataset={self.dataset},\n"
            f"    size={len(self.data)},\n"
            f"    split={self.split}\n"
            ")"
        )


@define
class LazyJsonDataset(Dataset):
    """Lazily load the json house data."""

    cached_data: dict = field(init=False)

    def __attrs_post_init__(self):
        self.cached_data = {}

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        if index not in self.cached_data:
            self.cached_data[index] = json.loads(self.data[index])
        return self.cached_data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return super().__len__()

    def __repr__(self):
        """Return a string representation of the dataset."""
        return super().__repr__()

    def __str__(self):
        """Return a string representation of the dataset."""
        return super().__str__()

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            if i not in self.cached_data:
                self.cached_data[i] = json.loads(x)
            yield self.cached_data[i]


@define
class DatasetDict:
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def __getitem__(self, key: str) -> Dataset:
        """Return the dataset with the given split."""
        if key == "train":
            if self.train is None:
                raise KeyError(key)
            return self.train
        elif key == "val":
            if self.val is None:
                raise KeyError(key)
            return self.val
        elif key == "test":
            if self.test is None:
                raise KeyError(key)
            return self.test
        else:
            raise KeyError(key)
