# Model package initialization
from .Model import Model, Model_legacy

# Make Model class available at package level
__all__ = ['BaseModel', 'Model', 'Model_legacy']
