from typing import Any, List, Literal, Optional

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
            f"DatasetSplit("
            f"    dataset={self.dataset},"
            f"    size={len(self.data)},"
            f"    split={self.split}"
            f")"
        )


@define
class DatasetDict:
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def __getitem__(self, key: str) -> Any:
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
