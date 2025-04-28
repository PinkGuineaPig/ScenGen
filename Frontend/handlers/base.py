import requests

API_BASE = "http://localhost:5000/api"

def fetch_all_model_configs():
    """
    GET /model-configs
    Fetch the complete list of all model configurations from the backend.
    Returns an empty list on failure.
    """
    try:
        resp = requests.get(f"{API_BASE}/model-configs")
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def fetch_som_configs(model_cfg_id: int):
    """
    GET /model-configs/{model_cfg_id}/som-configs
    Fetch all SOM configurations for a given model_config.
    Returns an empty list on failure.
    """
    try:
        resp = requests.get(f"{API_BASE}/model-configs/{model_cfg_id}/som-configs")
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def fetch_pca_configs(model_cfg_id: int):
    """
    GET /model-configs/{model_cfg_id}/pca-configs
    Fetch all PCA configurations for a given model_config.
    Returns an empty list on failure.
    """
    try:
        resp = requests.get(f"{API_BASE}/model-configs/{model_cfg_id}/pca-configs")
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


class BaseConfigHandler:
    """
    Abstract base class for modal‐specific handlers.

    Subclasses must define:
      - id_prefix: str, e.g. 'model', 'som', 'pca'
      - handle(trigger: str, **state) -> dict

    The return dict may include:
      - 'table':    new table data list
      - 'dialog':   bool to show ConfirmDialog
      - 'message':  str for ConfirmDialog
      - 'clear':    list of selected_rows to reset

    If no action is taken, handle() should return None or {}.
    """
    id_prefix: str = ""

    @classmethod
    def handles_trigger(cls, trigger: str) -> bool:
        """
        Return True if this handler is responsible for the given trigger id.
        """
        return trigger.startswith(f"{cls.id_prefix}-")

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        """
        Process the action for this handler.

        Args:
          trigger: the id of the component that fired (without prop)
          state:   keyword args for the relevant State values

        Returns:
          A dict with any of the keys 'table', 'dialog', 'message', 'clear'.
        """
        raise NotImplementedError("Subclasses must implement handle().")
