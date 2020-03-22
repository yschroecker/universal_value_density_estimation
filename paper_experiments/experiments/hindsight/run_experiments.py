import sacred
from sacred.observers import file_storage

from paper_experiments.experiments.hindsight import fetch_her, fetch_uvd, fetch_td3

noisy_slide_config = "iclr_experiments/experiments/hindsight/noisy_slide_config.json"
deterministic_slide_config = "iclr_experiments/experiments/hindsight/deterministic_slide_config.json"
deterministic_push_config = "iclr_experiments/experiments/hindsight/deterministic_push_config.json"


def _run_experiment(experiment: sacred.Experiment, config: str, observer: str, num_trials: int=1):
    experiment.add_config(config)
    experiment.observers.append(file_storage.FileStorageObserver.create(observer))
    for trial in range(num_trials):
        experiment.run()


def _main():
    _run_experiment(fetch_uvd.experiment, noisy_slide_config, "iclr_experiments/data/sacred/noisy_slide_uvd_924")
    _run_experiment(fetch_uvd.experiment, deterministic_slide_config,
                    "iclr_experiments/data/sacred/deterministic_slide_uvd_924")
    _run_experiment(fetch_uvd.experiment, deterministic_push_config,
                    "iclr_experiments/data/sacred/deterministic_push_uvd_924")
    _run_experiment(fetch_td3.experiment, deterministic_push_config,
                    "iclr_experiments/data/sacred/deterministic_push_td3_924")
    _run_experiment(fetch_td3.experiment, noisy_slide_config, "iclr_experiments/data/sacred/noisy_slide_td3_924")
    _run_experiment(fetch_td3.experiment, deterministic_slide_config,
                    "iclr_experiments/data/sacred/deterministic_slide_td3_924")
    _run_experiment(fetch_her.experiment, noisy_slide_config, "iclr_experiments/data/sacred/noisy_slide_her_924")
    _run_experiment(fetch_her.experiment, deterministic_slide_config,
                    "iclr_experiments/data/sacred/deterministic_slide_her_924")
    _run_experiment(fetch_her.experiment, deterministic_push_config,
                    "iclr_experiments/data/sacred/deterministic_push_her_924")


if __name__ == '__main__':
    _main()

