import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class DTSENDummyConfig:
    input_path: Path = Path(
        "metadata_sample/reconciled_sample_metadata.json"
    )

    output_path: Path = Path(
        "metadata_sample/reconciled_sample_metadata_with_dtsen.json"
    )

    seed: int = 42

    same_probability: float = 0.7


class DTSENDummyGenerator:

    LABELS = {
        "atap": [
            "Beton",
            "Genteng",
            "Seng",
            "Asbes",
            "Bambu",
            "Kayu/sirap",
            "Jerami/ijuk/daun-daunan/rumbia",
            "Lainnya",
            "Tidak terdeteksi",
        ],

        "dinding": [
            "Tembok",
            "Plesteran anyaman bambu/kawat",
            "Kayu/papan/gypsum/GRC/calciboard",
            "Anyaman bambu",
            "Batang kayu",
            "Bambu",
            "Lainnya",
            "Tidak terdeteksi",
        ],

        "lantai": [
            "Marmer/granit",
            "Keramik",
            "Parket/vinil/karpet",
            "Ubin/tegel/teraso",
            "Kayu/papan",
            "Semen/bata merah",
            "Bambu",
            "Tanah",
            "Lainnya",
            "Tidak terdeteksi",
        ],
    }

    def __init__(self, config: DTSENDummyConfig):
        self.config = config
        random.seed(config.seed)

    @staticmethod
    def load_json(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: List[Dict[str, Any]], path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

    def generate_dtsen_label(
        self,
        actual: str,
        component: str,
    ) -> str:

        candidates = self.LABELS[component]

        if actual == "Tidak terdeteksi":

            if random.random() < self.config.same_probability:
                return "Tidak terdeteksi"

            other = [
                x for x in candidates
                if x != "Tidak terdeteksi"
            ]

            return random.choice(other)

        if random.random() < self.config.same_probability:
            return actual

        other = [
            x for x in candidates
            if x != actual
        ]

        return random.choice(other)

    @staticmethod
    def generate_status(
        actual: str,
        dtsen: str,
    ) -> str:

        if actual == "Tidak terdeteksi":
            return "Tidak teridentifikasi"

        if actual == dtsen:
            return "Sesuai"

        return "Tidak sesuai"

    def process_record(
        self,
        record: Dict[str, Any]
    ) -> Dict[str, Any]:

        actual = record["actual_label"]

        dtsen = {}
        status = {}

        for component in [
            "atap",
            "dinding",
            "lantai",
        ]:

            actual_value = actual.get(component)

            dtsen_value = self.generate_dtsen_label(
                actual_value,
                component,
            )

            dtsen[component] = dtsen_value

            status[component] = self.generate_status(
                actual_value,
                dtsen_value,
            )

        record["dtsen"] = dtsen
        record["status"] = status

        return record

    def run(self):

        data = self.load_json(
            self.config.input_path
        )

        output = []

        summary = {
            "Sesuai": 0,
            "Tidak sesuai": 0,
            "Tidak teridentifikasi": 0,
        }

        for record in data:

            updated = self.process_record(
                record
            )

            output.append(updated)

            for status in updated[
                "status"
            ].values():

                summary[status] += 1

        self.save_json(
            output,
            self.config.output_path,
        )

        return {
            "total_house": len(output),
            "output_path": str(
                self.config.output_path
            ),
            **summary,
        }