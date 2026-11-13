"""Command-line entry point: ``wakeword-lab <command> -c config.toml``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, WakeWordConfig, load_config

DEFAULT_CONFIG = "configs/hey_joy.toml"


def log(msg: str = "", **kwargs) -> None:
    print(msg, flush=True, **kwargs)


def _resolve_mic(args) -> tuple[int, int]:
    from .devices import choose_input_device, probe_loudest

    device = choose_input_device(args.device) if args.device else None
    if device is not None and args.channel is not None:
        return device, args.channel
    if args.probe:
        return probe_loudest(device, log)
    return (device if device is not None else choose_input_device()), (args.channel or 0)


def cmd_devices(args, config) -> None:
    from .devices import list_devices

    list_devices(log)


def cmd_synth(args, config: WakeWordConfig) -> None:
    from .synth import synthesize

    log(f"Synthesizing clips for '{config.phrase}' into {config.clips_dir}")
    synthesize(config, log)


def cmd_negatives(args, config) -> None:
    from .negatives import CORPORA, download_corpus, download_validation_features

    data_dir = Path(args.data_dir)
    for name in args.corpora or list(CORPORA):
        download_corpus(name, data_dir, log)
    if args.validation:
        path = download_validation_features(data_dir, log)
        log(f"  validation features: {path}")


def cmd_train(args, config: WakeWordConfig) -> None:
    from .dataset import build_windows
    from .features import make_extractor
    from .listen import enroll, load_enrollment
    from .synth import read_manifest
    from .train import train

    clips = read_manifest(config.clips_dir)
    if not clips:
        raise SystemExit(f"No clips in {config.clips_dir}. Run `wakeword-lab synth` first.")

    if args.enroll:
        device, channel = _resolve_mic(args)
        enroll_pos, enroll_neg = enroll(config, device, channel, takes=args.takes, log=log)
    else:
        enroll_pos, enroll_neg = load_enrollment(config)
        if enroll_pos:
            log(f"  using {len(enroll_pos)} saved enrollment takes from {config.enroll_dir}")

    log(f"Embedding {len(clips)} clips with openWakeWord features...")
    train_set, holdout = build_windows(config, clips, make_extractor(), enroll_pos, enroll_neg, log)
    X, y = train_set.arrays()
    X_val, y_val = holdout.arrays()
    log(f"  train windows {len(y)} (pos {int(y.sum())})   holdout windows {len(y_val)} (pos {int(y_val.sum())})")
    log("Training...")
    train(X, y, X_val, y_val, config.training, config.model_path, config.model_name, log)


def cmd_evaluate(args, config: WakeWordConfig) -> None:
    from .evaluate import evaluate

    evaluate(
        config,
        threshold=args.threshold,
        include_all=args.all,
        fp_features=Path(args.fp_features) if args.fp_features else None,
        log=log,
    )


def cmd_listen(args, config: WakeWordConfig) -> None:
    from .listen import listen

    device, channel = _resolve_mic(args)
    listen(config, device, channel, threshold=args.threshold, log=log)


def cmd_demo(args, config: WakeWordConfig) -> None:
    """Synthesize, (optionally) enroll, train, evaluate and listen in one go."""
    from .synth import read_manifest

    log("=" * 62)
    log(f"  {config.phrase}  -  live wake-word demo")
    log("=" * 62)
    log("\n[1/4] Synthetic clips")
    if read_manifest(config.clips_dir) and not args.resynth:
        log(f"  reusing clips in {config.clips_dir}")
    else:
        cmd_synth(args, config)
    log("\n[2/4] Training")
    if config.model_path.exists() and not (args.retrain or args.enroll):
        log(f"  reusing {config.model_path} (pass --retrain to rebuild)")
    else:
        cmd_train(args, config)
        log("\n[3/4] Evaluation")
        cmd_evaluate(argparse.Namespace(threshold=args.threshold, all=False, fp_features=None), config)
    log("\n[4/4] Listening")
    cmd_listen(args, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wakeword-lab",
        description="Train and run a custom openWakeWord model from synthetic speech.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=DEFAULT_CONFIG, help=f"TOML config (default: {DEFAULT_CONFIG})")

    mic = argparse.ArgumentParser(add_help=False)
    mic.add_argument("--device", help="input device index or name substring")
    mic.add_argument("--channel", type=int, help="input channel on multi-channel interfaces")
    mic.add_argument("--probe", action="store_true", help="listen on every input and pick the loudest")

    thr = argparse.ArgumentParser(add_help=False)
    thr.add_argument("--threshold", type=float, help="detection threshold 0-1 (default: from config)")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", parents=[common], help="list audio devices").set_defaults(
        func=cmd_devices, needs_config=False
    )
    sub.add_parser("synth", parents=[common], help="generate synthetic clips with Piper").set_defaults(func=cmd_synth)

    p = sub.add_parser("negatives", parents=[common], help="download background-audio corpora")
    p.add_argument("--data-dir", default="data/negatives")
    p.add_argument("--corpora", nargs="*", choices=["fma", "fsd50k"])
    p.add_argument("--validation", action="store_true", help="also fetch openWakeWord's validation features")
    p.set_defaults(func=cmd_negatives, needs_config=False)

    p = sub.add_parser("train", parents=[common, mic], help="embed clips and train the classifier")
    p.add_argument("--enroll", action="store_true", help="record a few takes of your own voice first")
    p.add_argument("--takes", type=int, default=5)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[common, thr], help="score holdout clips per phrase")
    p.add_argument("--all", action="store_true", help="score every clip, not just the holdout split")
    p.add_argument("--fp-features", help="(T, 96) .npy of background features for false accepts/hour")
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("listen", parents=[common, mic, thr], help="run live detection").set_defaults(func=cmd_listen)

    p = sub.add_parser("demo", parents=[common, mic, thr], help="synth + train + evaluate + listen")
    p.add_argument("--enroll", action="store_true", help="record your own voice before training")
    p.add_argument("--takes", type=int, default=5)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--resynth", action="store_true")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = None
    if getattr(args, "needs_config", True):
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            raise SystemExit(f"config error: {exc}") from exc
    try:
        args.func(args, config)
    except ModuleNotFoundError as exc:
        extra = _EXTRAS.get((exc.name or "").split(".")[0])
        if not extra:
            raise
        raise SystemExit(f"'{exc.name}' is not installed. Run: pip install -e \".[{extra}]\"") from exc


_EXTRAS = {
    "piper": "tts",
    "torch": "train",
    "onnx": "train",
    "openwakeword": "train,live",
    "onnxruntime": "train,live",
    "sounddevice": "live",
    "soxr": "live",
}


if __name__ == "__main__":
    main()
