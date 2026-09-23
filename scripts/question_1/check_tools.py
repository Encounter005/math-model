"""Report whether question-1 external tools are configured and callable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from transformers import AutoProcessor, WhisperForConditionalGeneration


@dataclass(frozen=True)
class ToolStatus:
    available: bool
    detail: str
    version: str | None = None


def version_arguments(name: str) -> tuple[tuple[str, ...], ...]:
    """Return the version-command form required by a configured tool."""
    return (("--version",), ("-version",))


def command_version(path: str, name: str) -> str | None:
    """Return the first line of a command's version output when supported."""
    for arguments in version_arguments(name):
        try:
            completed = subprocess.run(
                [path, *arguments], capture_output=True, check=False, text=True, timeout=10
            )
        except OSError:
            return None
        output = (completed.stdout or completed.stderr).splitlines()
        if completed.returncode == 0 and output:
            return output[0]
    return None


def check_whisper_model(model: str, cache_dir: Path) -> ToolStatus:
    """Verify that Whisper processor and weights are available in the local cache."""
    try:
        AutoProcessor.from_pretrained(model, cache_dir=str(cache_dir), local_files_only=True)
        WhisperForConditionalGeneration.from_pretrained(model, cache_dir=str(cache_dir), local_files_only=True)
    except OSError as error:
        return ToolStatus(False, f"model unavailable in {cache_dir}: {error}")
    return ToolStatus(True, str(cache_dir), model)


def check_tools(tools: dict[str, str]) -> dict[str, ToolStatus]:
    """Check configured command-based tools."""
    statuses: dict[str, ToolStatus] = {}
    for name, value in tools.items():
        resolved = os.path.expandvars(value)
        if not resolved or "$" in resolved:
            statuses[name] = ToolStatus(False, "not configured")
        else:
            path = shutil.which(resolved)
            version = command_version(path, name) if path is not None else None
            statuses[name] = ToolStatus(
                version is not None,
                (
                    path
                    if version is not None
                    else f"command could not run: {path}"
                    if path is not None
                    else f"command not found: {resolved}"
                ),
                version,
            )
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/question_1.yaml"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("artifacts/question_1/run_metadata.json")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    statuses = check_tools(config["tools"])
    asr_status = check_whisper_model(
        config["models"]["asr"], Path(config["paths"]["model_cache"])
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "python": sys.version,
                "tools": {name: asdict(status) for name, status in statuses.items()},
                "models": {"asr": asdict(asr_status)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, status in statuses.items():
        print(f"{name}: {'OK' if status.available else 'MISSING'} ({status.detail})")
    print(f"asr: {'OK' if asr_status.available else 'MISSING'} ({asr_status.detail})")
    ready = all(status.available for status in statuses.values()) and asr_status.available
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
