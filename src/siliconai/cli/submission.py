"""Submission helpers and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from siliconai.cli.config import Configuration
    from siliconai.cli.logging import Logger


def create_slurm_submission_script(
    config: Configuration,
    n_gpu: int,
    n_node: int,
) -> Path:
    """Create Slurm submission script."""
    output_dir = config.output_path / f"run_{config.run_number()}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_file = config.output_path / f"run_{config.run_number()}" / "submit.sh"
    output_log = config.output_path / f"run_{config.run_number()}" / "slurm.log"
    duration = 23  # TODO: make configurable

    with output_file.open("w") as file:
        file.write(
            "#!/bin/bash\n"
            "\n"
            "#SBATCH --partition=gpu\n"
            f"#SBATCH --nodes={n_node}\n"
            f"#SBATCH --gres=gpu:{n_gpu}\n"
            f"#SBATCH --ntasks-per-node={n_gpu}\n"
            "#SBATCH --cpus-per-task=32\n"
            "#SBATCH --mem=0\n"
            f"#SBATCH --time=0-{duration}:00:00\n"
            f"#SBATCH --output={output_log}\n"
            "\n"
            "# debugging flags\n"
            "# export NCCL_DEBUG=INFO\n"
            "# export PYTHONFAULTHANDLER=1\n"
            "\n"
            "source ./scripts/setup_modules.sh\n"
            "source ./.venv/bin/activate\n"
            "\n"
            f"srun siliconai train -c {config.location} --batch"
            f" --ngpu {n_gpu} --nnode {n_node}",
        )

    return output_file


def submit(logger: Logger, config: Configuration, n_gpu: int, n_node: int) -> None:
    """Submit a training job."""
    logger.info("Submitting %s", config.name)

    # setup run number
    logger.info("Run number %d", config.run_number(training=True))

    # make the submission script
    script = create_slurm_submission_script(config, n_gpu, n_node)
    logger.info('Created submission script in "%s"', script)
