import yaml
import os
import threading  # Keep for threading.Lock compatibility
from typing import Dict, Any
import logging

# Set up logging for parameter updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DynamicParameterLoader:
    """
    A class that loads parameters from local YAML files.
    Remote gist loading has been disabled.
    """
    
    def __init__(self, 
                 local_yaml_path: str, 
                 gist_url: str = None,  # Disabled - kept for compatibility
                 update_interval: float = 10.0):  # Disabled - kept for compatibility
        """
        Initialize the parameter loader.
        
        Args:
            local_yaml_path: Path to the local parameters_default.yaml file
            gist_url: (Disabled) URL to the GitHub gist containing training_parameters.yaml
            update_interval: (Disabled) How often to check for updates (in seconds)
        """
        self.local_yaml_path = local_yaml_path
        self.gist_url = gist_url  # Keep for compatibility but not used
        self.update_interval = update_interval  # Keep for compatibility but not used
        self._parameters = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._update_thread = None
        self._logger = logging.getLogger(__name__)
        
        # Load initial parameters
        self._load_local_parameters()
        # Removed gist update thread startup
    
    def _load_local_parameters(self):
        """Load parameters from the local YAML file."""
        try:
            with open(self.local_yaml_path, 'r') as file:
                local_params = yaml.safe_load(file)
                
            with self._lock:
                self._parameters = local_params.copy()
                
                # Override observed_image_size if ablation_study_fisheye_instead_of_multi_scale_maps is False
                if not self._parameters.get('ablation_study_fisheye_instead_of_multi_scale_maps', False):
                    self._parameters['observed_image_size'] = 32
                    self._logger.info(f"Overriding observed_image_size to 32 (ablation_study_fisheye_instead_of_multi_scale_maps is False)")
                
        except Exception as e:
            self._logger.error(f"Failed to load local parameters from {self.local_yaml_path}: {e}")
            raise
    
    def _fetch_remote_parameters(self) -> Dict[str, Any]:
        """Disabled: Fetch parameters from the GitHub gist with cache-busting."""
        # This method is disabled but kept for compatibility
        self._logger.info("Remote parameter fetching is disabled")
        return {}
    
    def _update_parameters(self):
        """Disabled: Update parameters by merging remote parameters with local ones."""
        # This method is disabled but kept for compatibility
        self._logger.debug("Parameter updates from remote sources are disabled")
        pass
    
    def _update_loop(self):
        """Disabled: Background thread loop for periodic parameter updates."""
        # This method is disabled but kept for compatibility
        self._logger.debug("Parameter update loop is disabled")
        pass
    
    def _start_update_thread(self):
        """Disabled: Start the background update thread."""
        # This method is disabled but kept for compatibility
        self._logger.info("Background parameter updates are disabled - using local parameters only")
        pass
    
    def get_parameter(self, key: str, default=None):
        """
        Get a parameter value by key.
        
        Args:
            key: Parameter name
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value
        """
        with self._lock:
            return self._parameters.get(key, default)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get a copy of all current parameters."""
        with self._lock:
            return self._parameters.copy()
    
    def stop_updates(self):
        """Stop the background update thread."""
        self._stop_event.set()
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join()

class DynamicParameters:
    """
    A parameter object that provides attribute access to dynamically updated parameters.
    """
    
    def __init__(self, loader: DynamicParameterLoader):
        self._loader = loader
    
    def __getattr__(self, name):
        """
        Get parameter value using attribute access.
        If the parameter doesn't exist, raise AttributeError.
        """
        value = self._loader.get_parameter(name)
        if value is None:
            # Check if it exists in the parameters at all
            all_params = self._loader.get_all_parameters()
            if name not in all_params:
                raise AttributeError(f"Parameter '{name}' not found")
        return value
    
    def __setattr__(self, name, value):
        """Override to prevent direct setting of attributes (except internal ones)."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set parameter '{name}' directly. Parameters are read-only.")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the parameters_default.yaml file
yaml_file_path = os.path.join(script_dir, 'parameters_default.yaml')

# Initialize the dynamic parameter loader
# This will load local parameters only (gist loading disabled)
parameter_loader = DynamicParameterLoader(
    local_yaml_path=yaml_file_path,
    gist_url=None,  # Disabled
    update_interval=10.0  # Disabled but kept for compatibility
)

# Create the parameter object that provides attribute access
para = DynamicParameters(parameter_loader)