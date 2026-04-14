"""
FESTIM model wrapper package.

Provides the :class:`Model` (FESTIM 2.0 API) and :class:`Model_legacy`
(FESTIM 1.4 API) classes that encapsulate geometry specification,
material assignment, boundary condition setup, solver execution, and
result export for tritium-transport simulations driven from YAML
configuration files.
"""
from .Model import Model, Model_legacy

# Make Model class available at package level
__all__ = ['BaseModel', 'Model', 'Model_legacy']
