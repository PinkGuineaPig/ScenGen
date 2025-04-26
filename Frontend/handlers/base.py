import requests

API_BASE = "http://localhost:5000/api"

def fetch_all_configs():
    """
    Fetch the complete list of all configurations (models, SOM, PCA) from the backend.
    Returns an empty list on failure.
    """
    try:
        resp = requests.get(f"{API_BASE}/configs")
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


class BaseConfigHandler:
    """
    Abstract base class for modal-specific handlers.

    Subclasses should define:
      - id_prefix: str, e.g. 'model', 'som', 'pca'
      - handle(trigger: str, **state) -> dict

    The return dict may include:
      - 'table': new table data list
      - 'dialog': bool indicating whether to show ConfirmDialog
      - 'message': str for the ConfirmDialog message
      - 'clear': list of selected_rows to reset

    If no action is taken, handle() should return None or {}.
    """
    id_prefix: str = ""

    @classmethod
    def handles_trigger(cls, trigger: str) -> bool:
        """
        Return True if this handler should process the given trigger id.
        """
        return trigger.startswith(f"{cls.id_prefix}-")

    @classmethod
    def handle(cls, trigger: str, **state) -> dict:
        """
        Process the action for this handler.

        Parameters:
        - trigger: the id of the component that fired (without prop)
        - state: keyword args for the relevant State values

        Returns a dict with any of the keys 'table', 'dialog', 'message', 'clear'.
        """
        raise NotImplementedError("Subclasses must implement the handle() method.")
