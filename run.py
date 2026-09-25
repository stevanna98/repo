"""Command-line coordinator. Full searches require an explicit opt-in."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "validate", "train", "analyze", "external", "report", "all"])
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--output", help="New output directory, or identical existing run to resume")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--allow-full-run", action="store_true", help="Explicitly authorize >100 candidate fits or real-data training")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs" / ".cache"))
    from normative_vae.config import load_config
    cfg = load_config(ROOT / args.config)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(cfg["threads"]))
    if args.device:
        cfg["device"] = args.device
    if args.output:
        cfg["output_dir"] = args.output
    fits = len(cfg["architectures"]) * len(cfg["densities"]) * cfg["cv"]["outer"] * cfg["cv"]["inner"] * cfg["cv"]["candidates"]
    if args.stage in {"train", "all"} and (fits > 100 or not cfg["synthetic"]) and not args.allow_full_run:
        parser.error(f"This is a research run ({fits} candidate fits). Add --allow-full-run only when intentionally launching it.")
    import torch
    from threadpoolctl import threadpool_limits
    from normative_vae.utils import device_for, prepare_run, write_json
    from normative_vae.synthetic import generate
    from normative_vae.data import load_data
    torch.set_num_threads(cfg["threads"])
    device = device_for(cfg["device"])
    data_dir, output = ROOT / cfg["data_dir"], (ROOT / cfg["output_dir"]).resolve()
    with threadpool_limits(limits=cfg["threads"]):
        if args.stage == "generate" or (args.stage == "all" and cfg["synthetic"]):
            if not cfg["synthetic"]:
                parser.error("generate requires synthetic: true and a distinct data directory")
            generate(cfg, data_dir)
            print(f"Synthetic dataset: {data_dir}", flush=True)
            if args.stage == "generate":
                return
        reference, external, rois, report = load_data(data_dir, cfg)
        if set(reference.ids) & set(external.ids):
            raise ValueError("Reference and external subject IDs overlap")
        from normative_vae.cross_validation import make_splits
        from normative_vae.graphs import adjacency_batch
        make_splits(reference.sex, cfg["cv"]["outer"], cfg["cv"]["inner"], cfg["seed"])
        report["reference"]["stratification_checked"] = True
        for density in cfg["densities"]:
            adjacency_batch(reference.fc, density)
        report["reference"]["positive_edge_densities_checked"] = cfg["densities"]
        prepare_run(cfg, data_dir, output, ROOT)
        write_json(output / "input_validation.json", report)
        print(f"{'SYNTHETIC SOFTWARE VERIFICATION' if cfg['synthetic'] else 'REAL DATA'} | {device} | {output}", flush=True)
        if args.stage in {"train", "all"}:
            from normative_vae.cross_validation import run_cv
            run_cv(cfg, reference, output, device)
        if args.stage in {"analyze", "all"}:
            from normative_vae.reporting import run_reference_analysis
            run_reference_analysis(cfg, reference, rois, output, device)
        if args.stage in {"external", "all"}:
            from normative_vae.external import run_external
            run_external(cfg, reference, external, rois, output, device)
        if args.stage in {"report", "all"}:
            from normative_vae.reporting import make_reports
            make_reports(cfg, output)
        print(f"Completed {args.stage}; artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
