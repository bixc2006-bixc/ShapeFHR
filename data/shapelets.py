from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class ShapeletEntry:
    values: np.ndarray
    cls: int
    ig: float
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    variable: int = 0
    has_real_ig: bool = False


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("shapelet", value.get("values", value.get("value")))
        if value is None:
            raise ValueError("shapelet dict has no shapelet/values/value field")
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("empty shapelet")
    return array


def _first(item: dict, keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return default


def _append(
    entries: list[ShapeletEntry],
    *,
    values: Any,
    cls: int,
    ig: float,
    start_idx: Any = None,
    end_idx: Any = None,
    variable: Any = 0,
    has_real_ig: bool = False,
) -> None:
    entries.append(
        ShapeletEntry(
            values=_to_array(values),
            cls=int(cls),
            ig=float(ig),
            start_idx=None if start_idx is None else int(start_idx),
            end_idx=None if end_idx is None else int(end_idx),
            variable=0 if variable is None else int(variable),
            has_real_ig=bool(has_real_ig),
        )
    )


def load_all_shapelets(path: str | Path) -> list[ShapeletEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as handle:
        obj = pickle.load(handle)

    entries: list[ShapeletEntry] = []

    if isinstance(obj, dict) and "shapelets" in obj:
        shapelets = obj["shapelets"]
        igs = obj.get("ig", obj.get("info_gain", obj.get("information_gain")))
        classes = obj.get("class", obj.get("classes", obj.get("label", obj.get("labels"))))
        starts = obj.get("start_idx", obj.get("start_indices", obj.get("start")))
        ends = obj.get("end_idx", obj.get("end_indices", obj.get("end")))
        variables = obj.get("variable", obj.get("variables"))

        def at(values, index, fallback):
            return values[index] if values is not None and index < len(values) else fallback

        for i, values in enumerate(shapelets):
            metadata = values if isinstance(values, dict) else {}
            cls = at(classes, i, _first(metadata, ["class", "cls", "label"],
                                        0 if i < len(shapelets) // 2 else 1))
            ig = at(igs, i, _first(metadata, ["ig", "info_gain", "information_gain"]))
            _append(
                entries,
                values=values,
                cls=cls,
                ig=ig if ig is not None else -1_000_000 - i,
                start_idx=at(starts, i, _first(metadata, ["start_idx", "start", "ps"])),
                end_idx=at(ends, i, _first(metadata, ["end_idx", "end", "pe"])),
                variable=at(variables, i, _first(metadata, ["variable", "var"], 0)),
                has_real_ig=ig is not None,
            )

    elif isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values()):
        for cls_key, items in obj.items():
            try:
                default_cls = int(cls_key)
            except (TypeError, ValueError):
                default_cls = 0

            for j, item in enumerate(items):
                if isinstance(item, dict):
                    raw = _first(item, ["shapelet", "values", "value"])
                    if raw is None:
                        continue
                    ig_raw = _first(item, ["ig", "info_gain", "information_gain"])
                    _append(
                        entries,
                        values=raw,
                        cls=int(_first(item, ["class", "cls", "label"], default_cls)),
                        ig=float(ig_raw) if ig_raw is not None else float(-1_000_000 - len(entries) - j),
                        start_idx=_first(item, ["start_idx", "start", "ps"]),
                        end_idx=_first(item, ["end_idx", "end", "pe"]),
                        variable=_first(item, ["variable", "var"], 0),
                        has_real_ig=ig_raw is not None,
                    )
                elif isinstance(item, tuple) and len(item) == 5:
                    waveform, ig, start, end, _ = item
                    _append(entries, values=waveform, cls=default_cls, ig=ig,
                            start_idx=start, end_idx=end, has_real_ig=True)
                else:
                    _append(
                        entries,
                        values=item,
                        cls=default_cls,
                        ig=float(-1_000_000 - len(entries) - j),
                    )

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                raw = _first(item, ["shapelet", "values", "value"])
                if raw is None:
                    continue
                ig_raw = _first(item, ["ig", "info_gain", "information_gain"])
                _append(
                    entries,
                    values=raw,
                    cls=int(_first(item, ["class", "cls", "label"], 0 if i < len(obj) // 2 else 1)),
                    ig=float(ig_raw) if ig_raw is not None else float(-1_000_000 - i),
                    start_idx=_first(item, ["start_idx", "start", "ps"]),
                    end_idx=_first(item, ["end_idx", "end", "pe"]),
                    variable=_first(item, ["variable", "var"], 0),
                    has_real_ig=ig_raw is not None,
                )
            else:
                _append(
                    entries,
                    values=item,
                    cls=0 if i < len(obj) // 2 else 1,
                    ig=float(-1_000_000 - i),
                )
    else:
        raise ValueError(f"unsupported shapelet format in {path}")

    if not entries:
        raise ValueError(f"no shapelets parsed from {path}")

    return entries


def select_shapelets(entries: list[ShapeletEntry], topk_per_class: int) -> list[ShapeletEntry]:
    k = int(topk_per_class)
    if k <= 0:
        raise ValueError("topk_per_class must be positive")

    cls0 = [entry for entry in entries if int(entry.cls) == 0]
    cls1 = [entry for entry in entries if int(entry.cls) == 1]
    if len(cls0) < k or len(cls1) < k:
        raise ValueError(
            f"not enough class-balanced shapelets: cls0={len(cls0)}, cls1={len(cls1)}, requested={k} each"
        )

    if any(entry.has_real_ig for entry in entries):
        cls0 = sorted(cls0, key=lambda x: x.ig, reverse=True)
        cls1 = sorted(cls1, key=lambda x: x.ig, reverse=True)

    selected = cls0[:k] + cls1[:k]
    if any(entry.has_real_ig for entry in selected):
        selected = sorted(selected, key=lambda x: x.ig, reverse=True)

    print(
        f"[shapelets] selected={len(selected)}, cls0={sum(e.cls == 0 for e in selected)}, "
        f"cls1={sum(e.cls == 1 for e in selected)}, topk_per_class={k}"
    )
    return selected


def load_shapelets(path: str | Path, topk_per_class: int) -> list[ShapeletEntry]:
    return select_shapelets(load_all_shapelets(path), topk_per_class)


def serialize_shapelets(entries: list[ShapeletEntry]) -> list[dict]:
    return [
        {
            "values": np.asarray(entry.values, dtype=np.float32),
            "cls": int(entry.cls),
            "ig": float(entry.ig),
            "start_idx": entry.start_idx,
            "end_idx": entry.end_idx,
            "variable": int(entry.variable),
            "has_real_ig": bool(entry.has_real_ig),
        }
        for entry in entries
    ]


def deserialize_shapelets(items: list[dict]) -> list[ShapeletEntry]:
    return [
        ShapeletEntry(
            values=np.asarray(item["values"], dtype=np.float32).reshape(-1),
            cls=int(item["cls"]),
            ig=float(item.get("ig", 0.0)),
            start_idx=item.get("start_idx"),
            end_idx=item.get("end_idx"),
            variable=int(item.get("variable", 0)),
            has_real_ig=bool(item.get("has_real_ig", False)),
        )
        for item in items
    ]
