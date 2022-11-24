from concurrent.futures import Future
from pathlib import Path
import sys
from typing import Union

from mcli.api.model.run import Run
from mcli.models.run_config import RunConfig
from mcli.api.runs import create_run


class FineTuneRun:
    """Encapsulates a fine-tuning MCloud Run.
    Tasks should subclass FineTuneRun and implement the get_config() method.
    Args:
        config (dict|RunConfig, Optional): MCloud config for the fine-tune run

    ```python
    ledgar_finetune = FineTuneRun(RunConfig(
        name="ledgar-ft-gpt2",
        image="mosaicml/composer:latest",
        ...
    ))
    ```
    """

    def __init__(
        self,
        config: Union[dict, RunConfig],
    ):
        if isinstance(config, dict):
            self.config = RunConfig.from_dict(config)
        else:
            self.config = config

    @classmethod
    def from_file(cls, file: Union[Path, str]):
        if isinstance(file, Path):
            config = RunConfig.from_file(file)
            return cls(config=config)
        elif isinstance(file, str):
            if file.endswith(".yaml") or file.endswith(".yml"):
                config = RunConfig.from_file(file)
            else:
                raise ValueError("If FineTuneRun.from_file is passed a string, it must be a path to a YAML config file")
            return cls(config=config)
        else:
            raise TypeError("FineTuneRun.from_file accepts str or Path objects")


    @property
    def job_name(self) -> str:
        """Config `name`, falls back to `run_name`, then defaults to class name."""
        if self.config.name is not None:
            return self.config.name
        return self.config.run_name if self.config.run_name else self.__class__.__name__

    def create_run(self, future: bool = False) -> Union[Run, Future]:
        return create_run(run=self.config, future=future)


if __name__ == "__main__":
    # run sequential or parallel?
    # this is sequential, use futures for parallel, .create_run(future=True)
    yamls_to_run = sys.argv[1:]
    if len(yamls_to_run) < 7:
        print("Pass a list of yaml files to main.py to run each fine-tuning lex-GLUE job")
    if len(yamls_to_run) == 0:
        print("no yaml files passed, using default yamls")
        print("NEED TO WRITE DEFAULT YAMLS, only ledgar is currently 'done' (hyperparams untested)")
        yamls_to_run = ['yamls/ledgar.yaml']
    run_configs = [FineTuneRun.from_file(f) for f in yamls_to_run]
    for run_conf in run_configs:
        run = run_conf.create_run()
        print(f'Launching run {run.name} with id {run.run_uid}')
