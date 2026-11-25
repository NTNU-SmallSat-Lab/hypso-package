# -*- coding: utf-8 -*-

# This file is part of hypso.
#
# hypso is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hypso is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hypso.  If not, see <http://www.gnu.org/licenses/>.

"""
A python package form NTNU HYPSO Hyperspectral Image Processing
"""

__author__ = "Cameron Penne <cameron.l.penne@ntnu.no>"
__credits__ = "Norwegian University of Science and Technology"

# Public API: hypso.open(path) -> immutable Product.
from .api import Product, open  # noqa: A004
from .io.product import read_product, write_product
from .sensors.hypso1 import get_hypso1_wavelengths
from .sensors.hypso2 import get_hypso2_wavelengths


def __getattr__(name):
    # Resolved lazily (git is queried on first access, not at import) and
    # from the code actually imported - see hypso.versioning for why the
    # build-time _version.py / installed metadata alone can be wrong.
    if name == "__version__":
        from .versioning import package_info
        return package_info("hypso")["version"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
