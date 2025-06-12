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
from collections import OrderedDict
from enum import Enum
from typing import Union

from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer
from pyworkflow.protocol import PointerParam, BooleanParam, LEVEL_ADVANCED, EnumParam, STEPS_PARALLEL
from pyworkflow.utils import Message
from tomo.objects import SetOfTiltSeries

logger = logging.getLogger(__name__)
# Form variables
IN_TS_SET = 'inTsSet'
RE_STACK_OUT_TS = 'reStackOutTsSet'
MODEL = 'model'

# Models for binary classification
SWIN_LARGE = 'swin large'
SWIN_TINY = 'swin tiny'
EFFICIENT_NET_B3 = 'EfficientNet_b3'
RESNET_50 = 'resnet50'

modelsBinaryCl = OrderedDict()
modelsBinaryCl[SWIN_LARGE] = 0,
modelsBinaryCl[SWIN_TINY] = 1,
modelsBinaryCl[EFFICIENT_NET_B3] = 2,
modelsBinaryCl[RESNET_50] = 3


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
    stepsExecutionMode = STEPS_PARALLEL

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
                      help='If set to No, the output tilt-series will be mark the bad tilt-images at metadata '
                           'level. Otherwise, a new binary will be generated for each tilt-series containing only '
                           'the good tilt-images.')
        form.addParam(MODEL, EnumParam,
                      display=EnumParam.DISPLAY_HLIST,
                      choices=list(modelsBinaryCl.keys()),
                      label='Choose a pre-trained model',
                      expertLevel=LEVEL_ADVANCED,
                      help='Pre-trained model that will be used for classifying images.')
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        tsSet = self._getTsSet()

    # --------------------------- UTILS functions -----------------------------
    def _getFormAttrib(self, attribName) -> Union[Pointer, BooleanParam, EnumParam, None]:
        return getattr(self, attribName, None)

    def _getFormValue(self, attribName: str) -> Union[bool, int, None]:
        attrib = self._getFormAttrib(attribName)
        return attrib if attrib is None else attrib.get()

    def _getTsSet(self,
                  returnPointer: bool = False) -> Union[SetOfTiltSeries, None]:
        tsPointer = self._getFormAttrib(IN_TS_SET)
        return tsPointer if returnPointer else tsPointer.get()

    # def _getFormValue(self,
    #                    attribName: str,
    #                    returnPointer: bool = False) -> Union[SetOfTiltSeries, bool, int, None]:
    #     attribVal = getattr(self, attribName, None)
    #     return attribVal if returnPointer else attribVal.get()