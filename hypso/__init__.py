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

__author__ = "Cameron Penne <cameron.l.penne@ntnu.no>, Alvaro Flores <alvaro.f.romero@ntnu.no>"
__credits__ = "Norwegian University of Science and Technology"

from .hypso import Hypso
from .hypso1 import Hypso1
from .hypso2 import Hypso2
from .download import download_nc_files
#from .DataArrayDict import DataArrayDict
#from .DataArrayValidator import DataArrayValidator


try:
    from ._version import __version__  # noqa
except:
    pass
