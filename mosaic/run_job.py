import sys

from omegaconf import OmegaConf as om, DictConfig

from mosaic.tasks import main


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg: DictConfig = om.merge(yaml_cfg, cli_cfg)  # type: ignore
    task = cfg.task
    main(task, cfg)