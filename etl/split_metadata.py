from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SplitConfig:
    input_path: Path = Path("metadata_sample/reconciled_sample_metadata_with_dtsen.json")
    output_dir: Path = Path("metadata_sample/splits_house_type_aware")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


class HouseTypeAwareIterativeStratifiedSplitter:
    """
    Split metadata with a two-level strategy:

    Level 1:
      Split by house_type groups so that:
      - multi
      - single_exterior_only
      - single_interior_only
      all appear across train/val/test as much as possible.

    Level 2:
      Inside each house_type group, use iterative stratification
      for:
      - actual_label.atap
      - actual_label.dinding
      - actual_label.lantai
      - label combination token
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()
        self.rng = random.Random(self.config.seed)

    def load_records(self) -> List[Dict[str, Any]]:
        path = self.config.input_path
        if not path.exists():
            raise FileNotFoundError(f"Input file tidak ditemukan: {path}")

        if path.suffix.lower() == ".jsonl":
            records: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, list):
            raise ValueError("Input JSON harus berupa list of records.")

        return obj

    def save_json(self, data: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_house_type(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None

        mapping = {
            "multi": "multi",
            "single_exterior_only": "single_exterior_only",
            "single_interior_only": "single_interior_only",
            "exterior_only": "single_exterior_only",
            "interior_only": "single_interior_only",
        }
        return mapping.get(text, text)

    @staticmethod
    def _normalize_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text == "Tidak Terdeteksi":
            return "Tidak terdeteksi"
        return text

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record)
        rec["house_type"] = self._normalize_house_type(rec.get("house_type"))

        actual = rec.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        rec["actual_label"] = {
            "atap": self._normalize_label(actual.get("atap")),
            "dinding": self._normalize_label(actual.get("dinding")),
            "lantai": self._normalize_label(actual.get("lantai")),
        }
        return rec

    def _combo_key(self, record: Dict[str, Any], schema: str) -> str:
        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        atap = self._normalize_label(actual.get("atap")) or "None"
        dinding = self._normalize_label(actual.get("dinding")) or "None"
        lantai = self._normalize_label(actual.get("lantai")) or "None"

        if schema == "multi":
            return f"multi::{atap}||{dinding}||{lantai}"
        if schema == "single_exterior_only":
            return f"single_exterior_only::{atap}||{dinding}||{lantai}"
        if schema == "single_interior_only":
            return f"single_interior_only::{atap}||{dinding}||{lantai}"

        return f"unknown::{atap}||{dinding}||{lantai}"

    def _record_tokens(self, record: Dict[str, Any]) -> Set[str]:
        """
        Tokens used for iterative stratification inside a house_type group.

        We include:
        - atap
        - dinding
        - lantai
        - combo token
        """
        tokens: Set[str] = set()

        actual = record.get("actual_label", {})
        if not isinstance(actual, dict):
            actual = {}

        for comp in ("atap", "dinding", "lantai"):
            val = self._normalize_label(actual.get(comp))
            if val is not None:
                tokens.add(f"{comp}={val}")

        ht = self._normalize_house_type(record.get("house_type")) or "unknown"
        tokens.add(self._combo_key(record, ht))

        return tokens

    def _desired_split_sizes(self, n: int) -> Dict[str, int]:
        """
        Proportional split sizes, with guarantees:
        - n == 1 -> train only
        - n == 2 -> train + val
        - n >= 3 -> each split gets at least 1 if possible
        """
        if n <= 0:
            return {"train": 0, "val": 0, "test": 0}

        if n == 1:
            return {"train": 1, "val": 0, "test": 0}

        if n == 2:
            return {"train": 1, "val": 1, "test": 0}

        ratios = [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio]
        raw = [n * r for r in ratios]
        base = [int(math.floor(x)) for x in raw]
        remainder = n - sum(base)

        frac_order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
        for i in frac_order[:remainder]:
            base[i] += 1

        # ensure each split at least 1 if n >= 3
        for i in range(3):
            if base[i] == 0:
                base[i] = 1

        # rebalance if too many
        while sum(base) > n:
            idx = max(range(3), key=lambda i: base[i])
            if base[idx] > 1:
                base[idx] -= 1
            else:
                break

        # final safety
        while sum(base) < n:
            idx = max(range(3), key=lambda i: [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio][i])
            base[idx] += 1

        return {
            "train": base[0],
            "val": base[1],
            "test": base[2],
        }

    def _token_targets(self, token_counts: Counter) -> Dict[str, Dict[str, int]]:
        """
        Per-token target occurrence per split.

        Special rules:
        - count 1 -> train only
        - count 2 -> train + val
        - count 3 -> train + val + test
        - count > 3 -> proportional
        """
        targets: Dict[str, Dict[str, int]] = {}

        for token, count in token_counts.items():
            if count == 1:
                targets[token] = {"train": 1, "val": 0, "test": 0}
                continue
            if count == 2:
                targets[token] = {"train": 1, "val": 1, "test": 0}
                continue
            if count == 3:
                targets[token] = {"train": 1, "val": 1, "test": 1}
                continue

            ratios = [self.config.train_ratio, self.config.val_ratio, self.config.test_ratio]
            raw = [count * r for r in ratios]
            base = [int(math.floor(x)) for x in raw]
            remainder = count - sum(base)

            frac_order = sorted(range(3), key=lambda i: (raw[i] - base[i]), reverse=True)
            for i in frac_order[:remainder]:
                base[i] += 1

            while sum(base) > count:
                idx = max(range(3), key=lambda i: base[i])
                base[idx] -= 1

            targets[token] = {
                "train": base[0],
                "val": base[1],
                "test": base[2],
            }

        return targets


    def _sample_priority(
        self,
        sample_idx: int,
        sample_tokens: Dict[int, Set[str]],
        token_counts: Counter,
        remaining_targets: Dict[str, Dict[str, int]],
    ) -> float:
        tokens = sample_tokens[sample_idx]
        score = 0.0

        for token in tokens:
            support = max(token_counts[token], 1)
            remaining_total = sum(remaining_targets[token].values())
            rarity = 1.0 / support
            score += rarity * 2.0
            score += rarity * remaining_total

        return score

    def _choose_split(
        self,
        tokens: Set[str],
        split_sizes: Dict[str, int],
        current_sizes: Dict[str, int],
        remaining_targets: Dict[str, Dict[str, int]],
        token_counts: Counter,
    ) -> str:
        candidates = [s for s in self.SPLITS if current_sizes[s] < split_sizes[s]]
        if not candidates:
            return "train"

        best_split = candidates[0]
        best_score = float("-inf")

        for split in candidates:
            cap_left = split_sizes[split] - current_sizes[split]
            score = cap_left * 0.02

            for token in tokens:
                support = max(token_counts[token], 1)
                deficit = remaining_targets[token][split]
                weight = 5.0 if token_counts[token] <= 3 else 1.0
                score += weight * (deficit / support)

            # tie-break
            if split == "train":
                score += 0.03
            elif split == "val":
                score += 0.02
            else:
                score += 0.01

            if score > best_score:
                best_score = score
                best_split = split

        return best_split

    def _iterative_split_group(self, group_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        n = len(group_records)
        if n == 0:
            return {"train": [], "val": [], "test": []}

        split_sizes = self._desired_split_sizes(n)

        sample_tokens: Dict[int, Set[str]] = {}
        token_counts = Counter()
        token_to_samples: Dict[str, Set[int]] = defaultdict(set)

        for idx, rec in enumerate(group_records):
            tokens = self._record_tokens(rec)
            sample_tokens[idx] = tokens
            for token in tokens:
                token_counts[token] += 1
                token_to_samples[token].add(idx)

        remaining_targets = self._token_targets(token_counts)

        split_buckets: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        current_sizes = {"train": 0, "val": 0, "test": 0}
        unassigned: Set[int] = set(range(n))

        # anchor rare labels first (count 1/2/3)
        rare_tokens = sorted(token_counts.keys(), key=lambda t: (token_counts[t], t))
        for token in rare_tokens:
            if token_counts[token] > 3:
                continue

            preferred_splits = {
                1: ["train"],
                2: ["train", "val"],
                3: ["train", "val", "test"],
            }[token_counts[token]]

            for split in preferred_splits:
                if current_sizes[split] >= split_sizes[split]:
                    continue

                candidates = list(token_to_samples[token] & unassigned)
                if not candidates:
                    continue

                best_idx = max(
                    candidates,
                    key=lambda i: self._sample_priority(i, sample_tokens, token_counts, remaining_targets),
                )

                rec = dict(group_records[best_idx])
                rec["split"] = split
                split_buckets[split].append(rec)
                current_sizes[split] += 1

                for tok in sample_tokens[best_idx]:
                    if remaining_targets[tok][split] > 0:
                        remaining_targets[tok][split] -= 1

                unassigned.remove(best_idx)

        # iterative pass
        while unassigned:
            active_tokens = [t for t in rare_tokens if token_to_samples[t] & unassigned]
            if not active_tokens:
                idx = next(iter(unassigned))
                candidates = [s for s in self.SPLITS if current_sizes[s] < split_sizes[s]]
                chosen_split = max(candidates, key=lambda s: split_sizes[s] - current_sizes[s]) if candidates else "train"
            else:
                target_token = active_tokens[0]  # rarest active token
                candidate_indices = list(token_to_samples[target_token] & unassigned)

                idx = max(
                    candidate_indices,
                    key=lambda i: self._sample_priority(i, sample_tokens, token_counts, remaining_targets),
                )

                chosen_split = self._choose_split(
                    tokens=sample_tokens[idx],
                    split_sizes=split_sizes,
                    current_sizes=current_sizes,
                    remaining_targets=remaining_targets,
                    token_counts=token_counts,
                )

            rec = dict(group_records[idx])
            rec["split"] = chosen_split
            split_buckets[chosen_split].append(rec)
            current_sizes[chosen_split] += 1

            for tok in sample_tokens[idx]:
                if remaining_targets[tok][chosen_split] > 0:
                    remaining_targets[tok][chosen_split] -= 1

            unassigned.remove(idx)

        return split_buckets


    def split(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {
            "multi": [],
            "single_exterior_only": [],
            "single_interior_only": [],
        }

        for rec in records:
            ht = self._normalize_house_type(rec.get("house_type"))
            if ht not in groups:
                continue
            groups[ht].append(self._normalize_record(rec))

        final_buckets: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # split each house_type separately, then merge
        for house_type, group_records in groups.items():
            shuffled = group_records[:]
            self.rng.shuffle(shuffled)

            group_split = self._iterative_split_group(shuffled)

            for split in self.SPLITS:
                final_buckets[split].extend(group_split[split])

        # final shuffle in each split
        for split in self.SPLITS:
            self.rng.shuffle(final_buckets[split])

        return final_buckets


    def _extract_global_label_counter(self, records: List[Dict[str, Any]], comp: str) -> Counter:
        counter = Counter()
        for rec in records:
            actual = rec.get("actual_label", {})
            if not isinstance(actual, dict):
                continue
            val = actual.get(comp)
            if val is not None:
                counter[str(val)] += 1
        return counter

    def _extract_combo_counter(self, records: List[Dict[str, Any]], schema: Optional[str] = None) -> Counter:
        counter = Counter()
        for rec in records:
            actual = rec.get("actual_label", {})
            if not isinstance(actual, dict):
                continue

            atap = actual.get("atap")
            dinding = actual.get("dinding")
            lantai = actual.get("lantai")

            key = f"{atap}||{dinding}||{lantai}"

            if schema is None:
                counter[key] += 1
            else:
                ht = self._normalize_house_type(rec.get("house_type")) or "unknown"
                if ht == schema:
                    counter[key] += 1

        return counter

    def summarize(self, split_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        split_sizes = {s: len(split_buckets[s]) for s in self.SPLITS}

        house_type_dist = {
            split: dict(Counter(rec.get("house_type", "unknown") for rec in split_buckets[split]))
            for split in self.SPLITS
        }

        global_label_dist = {}
        for comp in ("atap", "dinding", "lantai"):
            global_label_dist[comp] = {
                split: dict(self._extract_global_label_counter(split_buckets[split], comp))
                for split in self.SPLITS
            }

        per_schema_label_dist: Dict[str, Dict[str, Dict[str, int]]] = {
            "multi": {},
            "single_exterior_only": {},
            "single_interior_only": {},
        }
        for schema in per_schema_label_dist.keys():
            schema_records_by_split = {
                split: [r for r in split_buckets[split] if self._normalize_house_type(r.get("house_type")) == schema]
                for split in self.SPLITS
            }
            for comp in ("atap", "dinding", "lantai"):
                per_schema_label_dist[schema][comp] = dict(
                    Counter(
                        str(rec.get("actual_label", {}).get(comp))
                        for split in self.SPLITS
                        for rec in schema_records_by_split[split]
                        if isinstance(rec.get("actual_label", {}), dict) and rec.get("actual_label", {}).get(comp) is not None
                    )
                )

        global_combo_dist = {
            split: dict(self._extract_combo_counter(split_buckets[split], schema=None))
            for split in self.SPLITS
        }

        per_schema_combo_dist: Dict[str, Dict[str, Dict[str, int]]] = {
            "multi": {},
            "single_exterior_only": {},
            "single_interior_only": {},
        }
        for schema in per_schema_combo_dist.keys():
            per_schema_combo_dist[schema] = {
                split: dict(
                    Counter(
                        f"{rec.get('actual_label', {}).get('atap')}||{rec.get('actual_label', {}).get('dinding')}||{rec.get('actual_label', {}).get('lantai')}"
                        for rec in split_buckets[split]
                        if self._normalize_house_type(rec.get("house_type")) == schema
                    )
                )
                for split in self.SPLITS
            }

        return {
            "split_sizes": split_sizes,
            "house_type_distribution": house_type_dist,
            "label_distribution_global": global_label_dist,
            "label_distribution_by_schema": per_schema_label_dist,
            "combo_distribution_global": global_combo_dist,
            "combo_distribution_by_schema": per_schema_combo_dist,
        }

    def run(self) -> Dict[str, Any]:
        records = self.load_records()
        split_buckets = self.split(records)

        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.json"
        val_path = out_dir / "val.json"
        test_path = out_dir / "test.json"
        all_path = out_dir / "all_with_split.json"

        self.save_json(split_buckets["train"], train_path)
        self.save_json(split_buckets["val"], val_path)
        self.save_json(split_buckets["test"], test_path)

        all_records = split_buckets["train"] + split_buckets["val"] + split_buckets["test"]
        self.save_json(all_records, all_path)

        summary = self.summarize(split_buckets)

        return {
            "total_records": len(records),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "all_path": str(all_path),
            **summary,
        }