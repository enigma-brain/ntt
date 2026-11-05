import warnings

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig

from ntt import settings
from ntt.utils import RankedLogger, log_config_tree_debug, print_config_tree

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path=f"{root}/config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main entry point for NTT application.

    Loads configuration from Hydra, instantiates MainSettings (which enables
    environment variable overrides), and executes the job.

    Args:
        cfg (DictConfig): Configuration from Hydra.

    Environment Variables:
        See MainSettings in settings.py for available overrides.
        Example: NTT_JOB__SEED=42 to override random seed.

    Config Flags:
        print_cfg=true: Print configuration tree as YAML
        print_schema=true: Print MainSettings JSON schema
        ignore_warnings=true: Disable Python warnings

    Examples:
        ntt                           # Run with default config (train job)
        ntt job=train                 # Run training job explicitly
        ntt print_cfg=true            # Print config tree
        ntt print_schema=true job=null  # Print schema without running job
        ntt job=null print_cfg=true   # Print config without running job
        NTT_JOB__SEED=42 ntt         # Override seed via env var
    """
    # Instantiate MainSettings from Hydra config
    # This creates the Settings object with all nested configurations
    # WARNING: instantiate() can execute arbitrary code from config files.
    # Only use with trusted configs. See: https://hydra.cc/docs/advanced/instantiate_objects/overview/
    setting: settings.MainSettings = instantiate(cfg)

    # Extras: filter warnings + print config tree
    if setting.ignore_warnings:
        log.warning("Disabling python warnings!")
        warnings.filterwarnings("ignore")
    # Print schema
    if setting.print_schema:
        schema = settings.MainSettings.model_json_schema()
        omegaconf_schema = DictConfig(schema)
        print_config_tree(omegaconf_schema, format="json", intro_str="MainSettings JSON Schema:")
    if setting.print_cfg:
        log_config_tree_debug(cfg, logger=log, intro_str="Hydra config")

    # Exit if no job is configured
    if setting.job is None:
        log.info("No job configured. Exiting.")
        return

    # Allow job to process the full config if needed
    setting.job.process_config(cfg)

    # Run job in context
    with setting.job as runner:
        runner.run()
