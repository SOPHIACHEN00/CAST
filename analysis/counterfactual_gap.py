"""Compute an awarded-minus-true contribution gap for an individual run.

Both input CSVs use RunID (the awarded run), AttackerID, Seed, PartitionSHA256,
InitialStateSHA256, ScheduleSHA256, TrainingProtocolSHA256, UtilityProtocolSHA256
and UtilityUnit=accuracy_fraction. Digests must identify the same original
client-to-shard mapping, initial state, realized honest-round schedule, training
protocol and evaluation protocol (including held-out data and final checkpoint
rule). For the excluded world the partition and schedule identify its paired
reference design; that world omits the designated client throughout.

The run index adds ContributionsCSV, ContributionsSHA256, DataAccess,
PrivateFitCount, AccessEvidence and AccessEvidenceSHA256. AccessEvidence names
the local audit record supporting the declared data access and private fit
count. Optional SettingJSON is a JSON object of run-key values
to select one experiment from a merged CSV (e.g. {"TaskID":"task-1"}).
DataAccess is either no_private_data or intermittent_private_training.

The control CSV adds World, WorldRunID and FinalUtility. Each intermittent run
needs two recorded worlds: honest_only_intermittent (participates only on its
realized honest rounds) and excluded. File paths are relative to the run index.
Matching checks the supplied provenance declarations; it cannot reconstruct
missing controls or authenticate the underlying experiments.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "Analysis Code"))
from contribution_csv import parse_bool, read_contribution_runs


DIGESTS = ["PartitionSHA256", "InitialStateSHA256", "ScheduleSHA256", "TrainingProtocolSHA256", "UtilityProtocolSHA256"]
MATCH = ["RunID", "AttackerID", "Seed", *DIGESTS, "UtilityUnit"]
RUN_FIELDS = MATCH + ["ContributionsCSV", "ContributionsSHA256", "DataAccess", "PrivateFitCount",
                      "AccessEvidence", "AccessEvidenceSHA256"]
CONTROL_FIELDS = MATCH + ["World", "WorldRunID", "FinalUtility"]
WORLDS = {"honest_only_intermittent", "excluded"}


def read_index(path, required):
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if missing := set(required).difference(table):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for name in required:
        table[name] = table[name].str.strip()
        if table[name].eq("").any():
            raise ValueError(f"{path}: blank {name}")
    for name in (column for column in required if column.endswith("SHA256")):
        if not table[name].str.fullmatch(r"[a-fA-F0-9]{64}").all():
            raise ValueError(f"{path}: invalid {name}")
        table[name] = table[name].str.lower()
    for name in (column for column in ("Seed", "AttackerID", "PrivateFitCount") if column in required):
        if not table[name].str.fullmatch(r"0|[1-9]\d*").all():
            raise ValueError(f"{path}: {name} must be a nonnegative integer")
    if not table["UtilityUnit"].eq("accuracy_fraction").all():
        raise ValueError(f"{path}: the supported Shapley utility is accuracy_fraction")
    return table


def sha256(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_awards(path):
    with path.open("rb") as source:
        fingerprint = sha256(source)
        source.seek(0)
        frame, keys = read_contribution_runs(source)
    return fingerprint, frame, keys


def awarded_run(index_row, cached, root):
    fingerprint, candidates, keys = cached((root / index_row["ContributionsCSV"]).resolve())
    if fingerprint != index_row["ContributionsSHA256"]:
        raise ValueError("ContributionsSHA256 disagrees with the actual contribution export")
    selector = json.loads(index_row.get("SettingJSON") or "{}")
    if not isinstance(selector, dict) or set(selector).difference(keys):
        raise ValueError("SettingJSON must use experiment-level keys from the contribution export")
    for name, value in selector.items():
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Expected a finite scalar selector for {name}")
        candidates = candidates.loc[candidates[name].eq(value)]
    if len(candidates[keys].drop_duplicates()) != 1:
        raise ValueError("Expected exactly one matching contribution run; check SettingJSON")
    identity = candidates.iloc[0]
    if identity["ContributionMethod"] != "shapley":
        raise ValueError("Expected raw exact Shapley contributions")
    for name in MATCH:
        if name in candidates and not candidates[name].astype(str).eq(index_row[name]).all():
            raise ValueError(f"{name} disagrees with the contribution export")
    with (root / index_row["AccessEvidence"]).open("rb") as evidence:
        if sha256(evidence) != index_row["AccessEvidenceSHA256"]:
            raise ValueError("AccessEvidenceSHA256 disagrees with the data-access audit record")
    attacker = candidates.loc[candidates["ClientID"].eq(identity["AttackerID"]), "WithAttacker"].item()
    return identity, float(attacker), math.fsum(candidates["WithAttacker"])


def true_contribution(index_row, identity, controls):
    access = index_row["DataAccess"]
    if access == "no_private_data":
        if (str(identity["AttackMethod"]).lower() in {"synthetic_replacement", "syn"}
                or int(index_row["PrivateFitCount"]) != 0 or parse_bool(identity["UseLazy"])):
            raise ValueError("The declared data-free run conflicts with its recorded method or schedule")
        if not controls.empty:
            raise ValueError("Unexpected control worlds for a declared data-free run")
        return 0.0, {}
    if (access != "intermittent_private_training" or int(index_row["PrivateFitCount"]) == 0
            or not parse_bool(identity["UseLazy"])):
        raise ValueError("Expected an explicitly evidenced data-access regime compatible with the run")
    if len(controls) != 2 or set(controls["World"]) != WORLDS:
        raise ValueError("An intermittent run requires both recorded worlds: honest_only_intermittent and excluded")
    for name in MATCH:
        if not controls[name].eq(index_row[name]).all():
            raise ValueError(f"The matched controls disagree on {name}")
    if not controls["WorldRunID"].is_unique or controls["WorldRunID"].eq(index_row["RunID"]).any():
        raise ValueError("The two control worlds and the awarded run must have distinct run IDs")
    by_world = controls.set_index("World")
    values = pd.to_numeric(by_world["FinalUtility"], errors="raise")
    if not values.between(0, 1).all():
        raise ValueError("Recorded final accuracies must be finite fractions in [0, 1]")
    return float(values["honest_only_intermittent"] - values["excluded"]), {
        "IntermittentWorldRunID": by_world.at["honest_only_intermittent", "WorldRunID"],
        "ExcludedWorldRunID": by_world.at["excluded", "WorldRunID"],
        "IntermittentUtility": values["honest_only_intermittent"], "ExcludedUtility": values["excluded"],
    }


def calculate(run_index, control_index=None, *, signed_share=False):
    runs = read_index(run_index, RUN_FIELDS)
    if runs.empty or not runs["RunID"].is_unique:
        raise ValueError("The run index must contain unique RunIDs and at least one run")
    controls = read_index(control_index, CONTROL_FIELDS) if control_index else pd.DataFrame(columns=CONTROL_FIELDS)
    if set(controls["RunID"]).difference(runs["RunID"]):
        raise ValueError("Control rows reference unknown awarded RunIDs")
    cached = lru_cache(maxsize=4)(read_awards)
    results = []
    for run in runs.to_dict("records"):
        try:
            identity, awarded, total = awarded_run(run, cached, Path(run_index).resolve().parent)
            actual, evidence = true_contribution(run, identity, controls.loc[controls["RunID"].eq(run["RunID"])])
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{run['RunID']}: {exc}") from exc
        gap = awarded - actual
        record = {**run, **{k: identity[k] for k in ("Dataset", "AttackMethod", "ClientNum", "AttackerID", "TrialID", "Alpha")},
                  **evidence, "AwardedRawSV": awarded, "TotalSignedSV": total,
                  "TrueContributionRaw": actual, "GapRaw": gap}
        if signed_share:
            record.update(ShareDenominator="awarded_run_total_signed_sv",
                          ShareStatus="valid" if total else "zero_denominator",
                          AwardedShare=awarded / total if total else None,
                          TrueContributionShare=actual / total if total else None,
                          GapShare=gap / total if total else None)
        results.append(record)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_index", type=Path)
    parser.add_argument("--controls", type=Path, help="Independent, matched final-utility exports for intermittent runs")
    parser.add_argument("--signed-share", action="store_true", help="Divide both terms by the signed total of the awarded run")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = calculate(args.run_index, args.controls, signed_share=args.signed_share)
        sources = {args.run_index.resolve()}
        if args.controls:
            sources.add(args.controls.resolve())
        for name in ("ContributionsCSV", "AccessEvidence"):
            sources.update((args.run_index.parent / path).resolve() for path in result[name].unique())
        if args.output.resolve() in sources:
            raise ValueError("Choose an output path distinct from the inputs")
        result.to_csv(args.output, index=False)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
