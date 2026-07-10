#!/usr/bin/env python3
"""Generate and validate M3D-C1 RMP-response scenario trees for TCABR.

A "scenario" is a parameter scan over toroidal mode number (n), fluid model
(single_fluid/two_fluid), RMP drive current level (kAt), and RMP coil set
(CPL/CPM/CPU/IL/IM/IU), laid out as:

    n<mode>/<fluid_model>/coil_sets_<kAt>kAt/<coil_set>_set_<index>/

See scenario.json (at the root of a generated tree) for the config that
produced it, and ../CLAUDE.md for the directory/file conventions.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = TOOLS_DIR / "templates"
REFERENCE_DIR = TOOLS_DIR / "reference_data"

FLUID_PARAMS = {
    "single_fluid": {"ipres": 0, "db_fac": "0.0", "job_code": "1f"},
    "two_fluid": {"ipres": 1, "db_fac": "1.0", "job_code": "2f"},
}

DEFAULT_N_VALUES = [1, 3, 6, 9]
DEFAULT_FLUID_MODELS = ["single_fluid", "two_fluid"]
DEFAULT_CURRENT_LEVELS_KAT = [1]
DEFAULT_COIL_SETS = ["CPL", "CPM", "CPU", "IL", "IM", "IU"]
DEFAULT_SET_INDEX = "000"

CASE_FILES = [
    "C1input",
    "batch_script.response",
    "run",
    "make_links",
    "clear_links",
    "rmall",
    "rmp_coil.dat",
    "rmp_current.dat",
]
EXECUTABLE_FILES = {"run", "make_links", "clear_links", "rmall"}

RMALL_CONTENT = "rm -rf C1ke normcurv *out profiles* *.h5\n"

MAKE_LINKS_LINE_RE = re.compile(r"^ln -s \{shared_data_dir\}/(\S+) \.$", re.MULTILINE)


def linked_item_names():
    """Item basenames make_links.template links in, derived from the template
    itself so clear_links can never drift out of sync with make_links."""
    text = (TEMPLATES_DIR / "make_links.template").read_text()
    names = MAKE_LINKS_LINE_RE.findall(text)
    if not names:
        raise RuntimeError("could not parse any 'ln -s ... .' lines out of make_links.template")
    return names


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = ["shot", "shared_data_dir", "executable", "run", "slurm"]


def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise SystemExit(f"scenario config missing required key(s): {', '.join(missing)}")

    config.setdefault("n_values", DEFAULT_N_VALUES)
    config.setdefault("fluid_models", DEFAULT_FLUID_MODELS)
    config.setdefault("current_levels_kAt", DEFAULT_CURRENT_LEVELS_KAT)
    config.setdefault("coil_sets", DEFAULT_COIL_SETS)
    config.setdefault("set_index", DEFAULT_SET_INDEX)

    bad_fluids = set(config["fluid_models"]) - set(FLUID_PARAMS)
    if bad_fluids:
        raise SystemExit(f"unknown fluid_models in config: {sorted(bad_fluids)}")
    bad_coils = set(config["coil_sets"]) - set(DEFAULT_COIL_SETS)
    if bad_coils:
        raise SystemExit(f"unknown coil_sets in config: {sorted(bad_coils)}")

    return config


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def case_dir(output_dir, n, fluid, kat, coil_set, set_index):
    return output_dir / f"n{n}" / fluid / f"coil_sets_{kat}kAt" / f"{coil_set}_set_{set_index}"


def render_case_files(n, fluid, coil_set, config):
    fp = FLUID_PARAMS[fluid]
    job_name = f"n{n}_{fp['job_code']}_{coil_set}"

    c1input = (TEMPLATES_DIR / "C1input.template").read_text().format(
        ipres=fp["ipres"], db_fac=fp["db_fac"], ntor=n
    )
    batch_script = (TEMPLATES_DIR / "batch_script.response.template").read_text().format(
        nodes=config["slurm"]["nodes"],
        ntasks=config["slurm"]["ntasks"],
        partition=config["slurm"]["partition"],
        job_name=job_name,
        time_limit=config["slurm"]["time"],
        cluster_binary=config["executable"]["cluster_binary"],
        solver_flag=config["executable"]["solver_flag"],
    )
    run = (TEMPLATES_DIR / "run.template").read_text().format(
        ranks=config["run"]["ranks"], executable_path=config["executable"]["local_path"]
    )
    make_links = (TEMPLATES_DIR / "make_links.template").read_text().format(
        shared_data_dir=config["shared_data_dir"].rstrip("/")
    )
    clear_links = "".join(f"rm -rf {name}\n" for name in linked_item_names())

    rmp_coil = (REFERENCE_DIR / "rmp_coil.dat").read_text()
    rmp_current = (REFERENCE_DIR / "rmp_current" / f"{coil_set}.dat").read_text()

    return {
        "C1input": c1input,
        "batch_script.response": batch_script,
        "run": run,
        "make_links": make_links,
        "clear_links": clear_links,
        "rmall": RMALL_CONTENT,
        "rmp_coil.dat": rmp_coil,
        "rmp_current.dat": rmp_current,
    }


def iter_axis_combinations(config):
    for n in config["n_values"]:
        for fluid in config["fluid_models"]:
            for kat in config["current_levels_kAt"]:
                for coil_set in config["coil_sets"]:
                    yield n, fluid, kat, coil_set


def cmd_init(args):
    config = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created, skipped, forced = 0, 0, 0
    for n, fluid, kat, coil_set in iter_axis_combinations(config):
        d = case_dir(output_dir, n, fluid, kat, coil_set, config["set_index"])
        exists = d.exists()
        if exists and not args.force:
            print(f"skip (exists): {d.relative_to(output_dir)}")
            skipped += 1
            continue

        d.mkdir(parents=True, exist_ok=True)
        files = render_case_files(n, fluid, coil_set, config)
        for name, content in files.items():
            path = d / name
            path.write_text(content)
            if name in EXECUTABLE_FILES:
                path.chmod(0o755)

        if exists:
            print(f"regenerated (--force): {d.relative_to(output_dir)}")
            forced += 1
        else:
            print(f"created: {d.relative_to(output_dir)}")
            created += 1

    (output_dir / "scenario.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\n{created} created, {forced} regenerated, {skipped} skipped. "
          f"scenario.json written to {output_dir}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

CASE_DIR_RE = re.compile(
    r"^n(?P<n>\d+)/(?P<fluid>[^/]+)/coil_sets_(?P<kat>[\d.]+)kAt/(?P<coil_set>[A-Za-z]+)_set_(?P<set_index>\d+)$"
)


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def discover_cases(root):
    """Yield (relpath, match) for directories that look like case dirs, and
    separately flag ones that are close but don't match the naming convention."""
    cases = []
    malformed = []
    for candidate in sorted(root.glob("n*/*/coil_sets_*kAt/*_set_*")):
        if not candidate.is_dir():
            continue
        rel = candidate.relative_to(root).as_posix()
        m = CASE_DIR_RE.match(rel)
        if m:
            cases.append((candidate, rel, m))
        else:
            malformed.append(rel)
    return cases, malformed


def cmd_validate(args):
    root = Path(args.path).resolve()
    issues = []

    cases, malformed = discover_cases(root)
    for rel in malformed:
        issues.append(f"{rel}: directory name doesn't match n<N>/<fluid>/coil_sets_<kAt>kAt/<COIL>_set_<idx> convention")

    if not cases:
        issues.append(f"no case directories found under {root}")

    ref_coil = (REFERENCE_DIR / "rmp_coil.dat").read_text()
    ref_current = {p.stem: p.read_text() for p in (REFERENCE_DIR / "rmp_current").glob("*.dat")}

    c1input_bodies = defaultdict(set)   # hash of file with ntor/ipres/db_fac lines stripped -> set of case rels
    job_names = defaultdict(list)       # job name -> [case rels]
    batch_bodies = defaultdict(set)     # hash of file with -J line stripped -> set of case rels
    static_file_hashes = {name: defaultdict(set) for name in ("run", "make_links", "clear_links", "rmall")}

    for path, rel, m in cases:
        fluid = m.group("fluid")
        n = m.group("n")
        coil_set = m.group("coil_set")

        if fluid not in FLUID_PARAMS:
            issues.append(f"{rel}: unrecognized fluid model directory '{fluid}'")
        if coil_set not in DEFAULT_COIL_SETS:
            issues.append(f"{rel}: unrecognized coil set '{coil_set}' (not one of {DEFAULT_COIL_SETS})")

        missing_files = [name for name in CASE_FILES if not (path / name).exists()]
        for name in missing_files:
            issues.append(f"{rel}: missing {name}")
        if missing_files:
            continue

        rmp_coil = (path / "rmp_coil.dat").read_text()
        if rmp_coil != ref_coil:
            issues.append(f"{rel}: rmp_coil.dat differs from reference geometry")

        if coil_set in ref_current:
            rmp_current = (path / "rmp_current.dat").read_text()
            if rmp_current != ref_current[coil_set]:
                issues.append(f"{rel}: rmp_current.dat differs from the reference pattern for {coil_set}")

        c1input = (path / "C1input").read_text()
        ntor_m = re.search(r"^\tntor = (\S+)$", c1input, re.MULTILINE)
        ipres_m = re.search(r"^\tipres = (\S+)", c1input, re.MULTILINE)
        dbfac_m = re.search(r"^\tdb_fac = (\S+)", c1input, re.MULTILINE)
        if not (ntor_m and ipres_m and dbfac_m):
            issues.append(f"{rel}: could not locate ntor/ipres/db_fac in C1input")
        else:
            if ntor_m.group(1) != n:
                issues.append(f"{rel}: C1input ntor={ntor_m.group(1)} does not match directory n{n}")
            if fluid in FLUID_PARAMS:
                expected_ipres = str(FLUID_PARAMS[fluid]["ipres"])
                expected_dbfac = FLUID_PARAMS[fluid]["db_fac"]
                if ipres_m.group(1) != expected_ipres:
                    issues.append(f"{rel}: C1input ipres={ipres_m.group(1)} does not match {fluid} (expected {expected_ipres})")
                if dbfac_m.group(1) != expected_dbfac:
                    issues.append(f"{rel}: C1input db_fac={dbfac_m.group(1)} does not match {fluid} (expected {expected_dbfac})")
            normalized = c1input.replace(ntor_m.group(0), "").replace(ipres_m.group(0), "").replace(dbfac_m.group(0), "")
            c1input_bodies[sha1(normalized)].add(rel)

        batch = (path / "batch_script.response").read_text()
        job_m = re.search(r"^#SBATCH -J (\S+)$", batch, re.MULTILINE)
        if not job_m:
            issues.append(f"{rel}: could not locate #SBATCH -J in batch_script.response")
        else:
            job_names[job_m.group(1)].append(rel)
            batch_bodies[sha1(batch.replace(job_m.group(0), ""))].add(rel)

        for name in static_file_hashes:
            static_file_hashes[name][sha1((path / name).read_text())].add(rel)

        for name in linked_item_names():
            link = path / name.rstrip("*") if "*" not in name else None
            if link is None:
                for match in path.glob(name):
                    if match.is_symlink() and not match.exists():
                        issues.append(f"{rel}: broken symlink {match.name}")
            elif link.is_symlink() and not link.exists():
                issues.append(f"{rel}: broken symlink {name}")

    if len(c1input_bodies) > 1:
        issues.append(
            "C1input bodies (excluding ntor/ipres/db_fac) differ across cases when they should be "
            f"identical: {len(c1input_bodies)} distinct versions found"
        )
    dup_jobs = {name: rels for name, rels in job_names.items() if len(rels) > 1}
    for name, rels in dup_jobs.items():
        issues.append(f"duplicate SLURM job name '{name}' used by {len(rels)} cases: {', '.join(rels)}")
    if len(batch_bodies) > 1:
        issues.append(
            f"batch_script.response contents (excluding -J) differ across cases: {len(batch_bodies)} distinct versions found"
        )
    for name, hashes in static_file_hashes.items():
        if len(hashes) > 1:
            issues.append(f"{name} differs across cases: {len(hashes)} distinct versions found (expected 1)")

    print(f"checked {len(cases)} case directories under {root}")
    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("no issues found.")
    return 0


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="generate (or extend) a scenario tree from a config")
    p_init.add_argument("--config", required=True, help="path to scenario.json")
    p_init.add_argument("--output-dir", required=True, help="root directory of the scenario tree")
    p_init.add_argument("--force", action="store_true",
                         help="re-render input files for cases that already exist "
                              "(never touches run.log, *.h5, symlinks, or other solver output)")
    p_init.set_defaults(func=cmd_init)

    p_validate = sub.add_parser("validate", help="check a scenario tree for consistency")
    p_validate.add_argument("--path", default=".", help="root directory of the scenario tree (default: .)")
    p_validate.set_defaults(func=lambda args: sys.exit(cmd_validate(args)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
