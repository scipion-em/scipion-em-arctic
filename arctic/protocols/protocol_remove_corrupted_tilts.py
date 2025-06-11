# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * National Center of Biotechnology, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import logging
from enum import Enum

from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.protocol import PointerParam, BooleanParam, LEVEL_ADVANCED
from pyworkflow.utils import Message
from tomo.objects import SetOfTiltSeries

logger = logging.getLogger(__name__)
# Form variables
IN_TS_SET = 'inTsSet'
RE_STACK_OUT_TS = 'reStackOutTsSet'


class arcticOutputs(Enum):
    tiltSeries = SetOfTiltSeries


class ProtArcticRemoveCorruptedTilts(EMProtocol):
    """Automated Removal of Corrupted Tilts in Cryo-ET. Corrupted tilt-images are the ones
    in which any of these effects/artifacts is detected: drift, ice reflection, lamella edge
    thick lamella, and contamination. More details in
    https://www.biorxiv.org/content/10.1101/2025.03.13.642992v1."""

    _label = 'automated removal of corrupted tilts'
    _devStatus = BETA
    _possibleOutputs = arcticOutputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      important=True,
                      label='Tilt-Series')
        form.addParam(RE_STACK_OUT_TS, BooleanParam,
                      default=False,
                      label='Re-stack the output tilt-series?',
                      expertLevel=LEVEL_ADVANCED,
                      help='If set to No, the output tilt-series will be mark the bad tilt-images at metadata '
                           'level. Otherwise, a new binary will be generated for each tilt-series containing only '
                           'the good tilt-images.')
