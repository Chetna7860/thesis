import os
import yaml
from typing import Dict, Any, Optional
from easydict import EasyDict as edict


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = edict()
        if config_path is not None:
            self.load(config_path)
        self._post_process()

    def load(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        self.cfg = edict(raw)

    def _post_process(self):
        for key in ['checkpoint_dir', 'log_dir', 'plot_dir', 'model_dir', 'gradcam_dir']:
            path_key = key
            if hasattr(self.cfg.output, key):
                setattr(self.cfg.output, key, os.path.abspath(getattr(self.cfg.output, key)))
                os.makedirs(getattr(self.cfg.output, key), exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.cfg, name):
            return getattr(self.cfg, name)
        raise AttributeError(f"Config has no attribute '{name}'")

    @property
    def device(self):
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_active_datasets(self):
        active = []
        if self.cfg.data.breakhis.use:
            active.append('breakhis')
        if self.cfg.data.inbreast.use:
            active.append('inbreast')
        if self.cfg.data.mias.use:
            active.append('mias')
        return active
