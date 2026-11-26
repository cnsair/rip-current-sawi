"""Matched SAWI reviewer control. Run from the root of rip-current-sawi."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

METRICS = ("recall", "f2", "miou")
ALPHAS = {"010": "0.10", "070": "0.70"}
SCRIPTS = (
    "wise_ft_interpolate.py", "bn_recalibrate.py", "evaluate_test_set.py",
    "dual_branch_segformer.py",
)
BN_COMMAND = (
    "import runpy, torch; torch.manual_seed(0); "
    "runpy.run_path('bn_recalibrate.py', run_name='__main__')"
)


def file_sha256(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, default=Path("trained_models/segformer_b2_local.pth"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("trained_models/determinism_check"))
    parser.add_argument("--train-images", type=Path, default=Path("data_local/train_local/images"))
    parser.add_argument("--test-images", type=Path, default=Path("data_local/test_local/rip_vis_val_images/images"))
    parser.add_argument("--test-masks", type=Path, default=Path("data_local/test_local/rip_vis_val_images/masks"))
    parser.add_argument("--out", type=Path, default=Path("reviewer_interpolation/full"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--with-epoch-swad", action="store_true",
                        help="Include trained_models/baselines_seedN/baseline_swadwin.pth")
    parser.add_argument("--with-epoch-swa", action="store_true",
                        help="Include trained_models/baselines_seedN/baseline_swa.pth")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and print checkpoint metadata")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Give each seed once")
    if args.batches < 1 or args.batch_size < 1:
        parser.error("Batch count and batch size must be positive")
    return args


def name_digest(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def checkpoint_metadata(path, torch):
    from collections import Counter

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    if not state:
        raise ValueError(f"Empty model_state in {path}")
    config = checkpoint.get("config", {})
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "epoch": checkpoint.get("epoch"),
        "amp_dtype": str(config.get("amp_dtype", "unrecorded")),
        "checkpoint_seed": config.get("seed"),
        "state_tensor_dtypes": dict(Counter(str(tensor.dtype) for tensor in state.values())),
        "dense_metadata": checkpoint.get("swad"),
        "validation_score": str(checkpoint.get("val_iou", "unrecorded")),
    }


def prepare(args):
    import torch

    segformer = Path("segformer-b2-local")
    files = [Path(name) for name in SCRIPTS] + [segformer / "config.json", args.anchor]
    parents = {}
    for seed in args.seeds:
        stem = args.checkpoint_dir / f"segformer_b2_ft_bf16_seed{seed}"
        parents[seed] = {
            "direct": stem.with_suffix(".pth"),
            "sawi": stem.with_name(stem.name + "_swad.pth"),
        }
        if args.with_epoch_swad:
            parents[seed]["epoch_swad"] = Path(f"trained_models/baselines_seed{seed}/baseline_swadwin.pth")
        if args.with_epoch_swa:
            parents[seed]["epoch_swa"] = Path(f"trained_models/baselines_seed{seed}/baseline_swa.pth")
        files.extend(parents[seed].values())
    hf_weights = [path for pattern in ("model*.safetensors", "pytorch_model*.bin")
                  for path in segformer.glob(pattern) if path.is_file()]
    if not hf_weights:
        raise ValueError(f"Missing local model weights under {segformer}")
    missing = [str(path) for path in files if not path.is_file()]
    missing += [str(path) for path in (args.train_images, args.test_images, args.test_masks) if not path.is_dir()]
    if missing:
        raise ValueError("Missing input files or directories:\n" + "\n".join(missing))

    # Mirror the repository loaders' extension rules.
    train_names = [path.name for path in args.train_images.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if len(train_names) < args.batches * args.batch_size:
        raise ValueError(f"Found {len(train_names)} training images, need {args.batches * args.batch_size}")
    transfer = [*args.test_images.glob("*.jpg"), *args.test_images.glob("*.png")]
    if len(transfer) != 4349 or len({p.stem for p in transfer}) != 4349:
        raise ValueError(f"Expected 4,349 unique RipVIS image stems, found {len(transfer)} image paths")
    missing_masks = [path.name for path in transfer if not (args.test_masks / (path.stem + ".png")).is_file()]
    if missing_masks:
        raise ValueError(f"Missing masks for {len(missing_masks)} images, e.g. {missing_masks[:5]}")
    expected_names = {path.name for path in transfer}

    manifest = {
        "seeds": args.seeds,
        "alphas": ALPHAS,
        "bn_batch_size": args.batch_size,
        "bn_batches": args.batches,
        "bn_seed": 0,
        "train_images": str(args.train_images.resolve()),
        "hf_model_weights": {str(path): file_sha256(path) for path in sorted(hf_weights)},
        "train_image_names_sha256": name_digest(train_names),
        "ripvis_images": str(args.test_images.resolve()),
        "ripvis_masks": str(args.test_masks.resolve()),
        "ripvis_image_names_sha256": name_digest(expected_names),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "scripts": {str(path): file_sha256(path) for path in [*map(Path, SCRIPTS), segformer / "config.json",
                                                               Path(__file__)]},
        "anchor": checkpoint_metadata(args.anchor, torch),
        "parents": {str(seed): {arm: checkpoint_metadata(path, torch) for arm, path in methods.items()}
                    for seed, methods in parents.items()},
    }
    print(json.dumps(manifest, indent=2), flush=True)
    return parents, manifest, expected_names


def model_scores(csv_path, expected_names):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = {row["image"] for row in rows}
    if names != expected_names or len(rows) != len(expected_names):
        raise ValueError(f"Unexpected image list or duplicate rows in {csv_path}")
    scores = {}
    for metric in METRICS:
        per_image = [float(row[metric]) for row in rows]
        if any(not 0 <= value <= 1 for value in per_image):
            raise ValueError(f"Invalid {metric} in {csv_path}")
        scores[metric] = statistics.fmean(per_image)
    return scores


def run_one(args, label, parent, expected_names, tag=None, calibrate=True):
    result_dir = args.out / "ripvis"
    result_csv = result_dir / f"{label}_per_image.csv"
    done = result_dir / f"{label}.done"
    if done.exists():
        print(f"Reusing completed evaluation: {label}", flush=True)
        return model_scores(result_csv, expected_names)

    work_dir = args.out / "models" / label
    work_dir.mkdir(parents=True, exist_ok=True)
    if tag is not None:
        subprocess.run([sys.executable, "wise_ft_interpolate.py", "--anchor", str(args.anchor),
                        "--finetuned", str(parent), "--outdir", str(work_dir),
                        "--alphas", ALPHAS[tag]], check=True)
        checkpoint = work_dir / f"segformer_b2_wise_a{tag}.pth"
    elif calibrate:
        checkpoint = work_dir / "model.pth"
        shutil.copyfile(parent, checkpoint)
    else:
        checkpoint = parent

    if calibrate:
        # One invocation per checkpoint resets both the data order and torch RNG.
        subprocess.run([sys.executable, "-c", BN_COMMAND, "--wise-dir", str(work_dir),
                        "--pattern", checkpoint.name, "--train-images", str(args.train_images),
                        "--batches", str(args.batches), "--batch-size", str(args.batch_size),
                        "--seed", "0", "--force"], check=True)

    env = dict(os.environ, EVAL_IMAGES=str(args.test_images), EVAL_MASKS=str(args.test_masks),
               EVAL_RESULTS=str(result_dir))
    subprocess.run([sys.executable, "evaluate_test_set.py", "--family", "segformer",
                    "--checkpoint", str(checkpoint), "--segformer-path", "./segformer-b2-local",
                    "--label", label], env=env, check=True)
    scores = model_scores(result_csv, expected_names)
    done.touch()
    return scores


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(args, records, by_arm):
    from scipy.stats import t

    write_csv(args.out / "per_run.csv", records, ["arm", "seed", *METRICS])
    spread = []
    for arm, runs in by_arm.items():
        for metric in METRICS:
            values = [run[metric] for run in runs.values()]
            spread.append(dict(arm=arm, metric=metric, n=len(values), mean=statistics.fmean(values),
                               min=min(values), max=max(values), range_pp=100 * (max(values) - min(values)),
                               sd_pp=100 * statistics.stdev(values) if len(values) > 1 else None))
    write_csv(args.out / "spread.csv", spread,
              ["arm", "metric", "n", "mean", "min", "max", "range_pp", "sd_pp"])

    paired = []
    for tag in ALPHAS:
        reference_arms = [arm for arm in ("direct", "epoch_swad", "epoch_swa")
                          if f"{arm}_a{tag}" in by_arm]
        for reference in reference_arms:
            for metric in METRICS:
                sawi = by_arm[f"sawi_a{tag}"]
                control = by_arm[f"{reference}_a{tag}"]
                diffs = [100 * (sawi[seed][metric] - control[seed][metric]) for seed in args.seeds]
                mean = statistics.fmean(diffs)
                n = len(diffs)
                radius = t.ppf(0.975, n - 1) * statistics.stdev(diffs) / math.sqrt(n) if n > 1 else None
                paired.append(dict(alpha=ALPHAS[tag], contrast=f"sawi - {reference}", metric=metric, n=n,
                                   mean_pp=mean, ci95_lo_pp=mean - radius if radius is not None else None,
                                   ci95_hi_pp=mean + radius if radius is not None else None))
    write_csv(args.out / "paired_effects.csv", paired,
              ["alpha", "contrast", "metric", "n", "mean_pp", "ci95_lo_pp", "ci95_hi_pp"])


def main():
    args = arguments()
    parents, manifest, expected_names = prepare(args)
    if args.check_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ripvis").mkdir(exist_ok=True)
    manifest_file = args.out / "manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Input manifest changed. Use a new --out directory.")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))
    records, by_arm = [], {}

    def evaluate(arm, seed, parent, tag=None, calibrate=True):
        label = f"{arm}_seed{seed}" if seed is not None else arm
        scores = run_one(args, label, parent, expected_names, tag, calibrate)
        by_arm.setdefault(arm, {})[seed] = scores
        records.append(dict(arm=arm, seed=seed, **scores))

    evaluate("anchor_native", None, args.anchor, calibrate=False)
    evaluate("anchor_matched_bn", None, args.anchor)
    for seed, methods in parents.items():
        evaluate("endpoint_matched_bn", seed, methods["direct"])
        evaluate("dense_matched_bn", seed, methods["sawi"])
        for arm, parent in methods.items():
            if arm in ("epoch_swad", "epoch_swa"):
                evaluate(f"{arm}_matched_bn", seed, parent)
            for tag in ALPHAS:
                evaluate(f"{arm}_a{tag}", seed, parent, tag)
    summarise(args, records, by_arm)
    print(f"Summary tables are in {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Experiment stopped: {error}")
