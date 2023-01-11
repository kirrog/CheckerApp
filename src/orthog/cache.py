import threading

from src.orthog.predict import ReorthPredictor


class ModelCache(object):
    """ """
    __shared_state = {
        "_model": None,
        "_lock": threading.Lock()
    }

    def __init__(self):
        self.__dict__ = self.__shared_state

    def load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # загружаем модель
                    self._model = ReorthPredictor('DeepPavlov/rubert-base-cased-sentence')

    @property
    def model(self) -> ReorthPredictor:
        self.load_model()
        return self._model
