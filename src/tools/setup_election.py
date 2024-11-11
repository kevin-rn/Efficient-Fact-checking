import json
import os
from itertools import chain
import uuid

import pandas as pd
import os

ELEC_FOLDER = os.path.join("data", "elections")

def convert_file() -> None:
    """
    Converts the election file into HoVer format.
    """
    query_path = os.path.join(ELEC_FOLDER, "trump_harris_claims_annotated_sean.csv")
    elec_df = pd.read_csv(query_path)

    hover_format_claims = []
    for index, row_obj in elec_df.iterrows():
        uid = str(uuid.uuid4())
        claim = row_obj['Cleaned']
        supporting_facts = [ ]
        label = (
            "SUPPORTED"
            if row_obj["CleanedVerdict"] == "Supported"
            else "NOT_SUPPORTED"
        )
        hover_format_claims.append(
            {
                "uid": uid,
                "claim": claim,
                "supporting_facts": supporting_facts,
                "label": label,
            }
        )

    json_path = os.path.join(ELEC_FOLDER, f"elec_dev_release_v1.1.json")
    with open(json_path, "w") as outfile:
        json.dump(hover_format_claims, outfile)

    # Cleanup
    # os.remove(query_path)


def main():
    """
    Converts elections file to HoVer claim data format.
    """
    convert_file()


if __name__ == "__main__":
    main()
