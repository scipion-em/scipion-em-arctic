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
from os.path import join
from typing import Union

from arctic import Plugin, ARCTIC_REPO_DIRNAME, MODELS_PARENT_DIR
from arctic.constants import BINARY_MODELS_DIR
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer
from pyworkflow.protocol import PointerParam, BooleanParam, LEVEL_ADVANCED, EnumParam, STEPS_PARALLEL, GPU_LIST, \
    StringParam
from pyworkflow.utils import Message, makePath
from tomo.objects import SetOfTiltSeries

logger = logging.getLogger(__name__)

# Form variables
IN_TS_SET = 'inTsSet'
RE_STACK_OUT_TS = 'reStackOutTsSet'
GEN_PDF_REPOS = 'genPdfRepos'
MODEL = 'model'

# Models for binary classification
SWIN_LARGE = 'swin large'
SWIN_TINY = 'swin tiny'
EFFICIENT_NET_B3 = 'EfficientNet_b3'
RESNET_50 = 'resnet50'

modelsBinaryCl = OrderedDict()
modelsBinaryCl[SWIN_LARGE] = 0
modelsBinaryCl[SWIN_TINY] = 1
modelsBinaryCl[EFFICIENT_NET_B3] = 2
modelsBinaryCl[RESNET_50] = 3

modelsBinaryClFn = {
    modelsBinaryCl[SWIN_LARGE]: 'swin_large_fine-tuned.pth',
    modelsBinaryCl[SWIN_TINY]: 'swin_tiny_fine-tuned.pth',
    modelsBinaryCl[EFFICIENT_NET_B3]: 'efficientnet_b3_fine-tuned.pth',
    modelsBinaryCl[RESNET_50]: 'resnet50_fine-tuned.pth'
}

# Other vars
PDF_REPORT_DIR = 'reports'


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
    program = 'run_TS_cleaning.py'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = None

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

        form.addParam(GEN_PDF_REPOS, BooleanParam,
                      default=True,
                      label='Generate a pdf report?',
                      help=f'A pdf report will be generated for each tilt-series in protocolFolder/extra/'
                           f'{PDF_REPORT_DIR}. Each report will contain the tilt angles and excluded images '
                           f'with the corresponding probability bars.')

        form.addParam(MODEL, EnumParam,
                      display=EnumParam.DISPLAY_COMBO,
                      choices=list(modelsBinaryCl.keys()),
                      default=modelsBinaryCl[SWIN_LARGE],
                      label='Choose a pre-trained model',
                      expertLevel=LEVEL_ADVANCED,
                      help=f'Pre-trained model that will be used for classifying images. '
                           f'Models available are (according to the paper '
                           f'https://www.biorxiv.org/content/10.1101/2025.03.13.642992v1) :\n'
                           f'\n\n- *{SWIN_LARGE}*: transformer-based (uses attention mechanisms). '
                           f'Excellent overall performance, especially in identifying clean (good) '
                           f'tilt images. It was the top performer in balancing precision and recall.'
                           f'\n\n- *{SWIN_TINY}*: lightweight Transformer. Good trade-off between speed '
                           f'and performance. good accuracy. Recommended for quick testing or use on '
                           f'systems with limited resources.'
                           f'\n\n- *{EFFICIENT_NET_B3}*: Convolutional Neural Network (CNN). Very effective '
                           f'at detecting corrupted tilts. Offers strong performance with moderate '
                           f'computational cost.'
                           f'\n\n- *{RESNET_50}*: CNN with residual connections. High precision in detecting '
                           f'corrupted tilts. Slightly conservative—may miss some corrupted tilts.'
                      )

        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs",
                       help='GPU device/s to be used.')
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for tsId in self.tsDict.keys():
            runId = self._insertFunctionStep(self.runArctic, tsId,
                                             prerequisites=[],
                                             needsGPU=True)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=runId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)

        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.tsDict = {ts.getTsId(): ts.clone() for ts in self._getTsSet()}
        if self._getFormValue(GEN_PDF_REPOS):
            makePath(self._getPdfReportDir())

    def runArctic(self, tsId: str):
        Plugin.runArctic(self, self._getProgram(), self._getArcticCmd(tsId))

    def createOutputStep(self, tsId: str):
        ts = self.tsDict[tsId]

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

    def _getProgram(self) -> str:
        return Plugin.getHome(*[ARCTIC_REPO_DIRNAME, self.program])

    def _getModelFile(self) -> str:
        chosenModel = self._getFormValue(MODEL)
        modelFn = modelsBinaryClFn[chosenModel]
        return Plugin.getHome(*[ARCTIC_REPO_DIRNAME,
                                MODELS_PARENT_DIR,
                                BINARY_MODELS_DIR,
                                modelFn])

    def _getPdfReportDir(self) -> str:
        return self._getExtraPath(PDF_REPORT_DIR)

    def _getPdfReportFn(self, tsId: str) -> str:
        return join(self._getPdfReportDir(), f'{tsId}.pdf')

    def _getOutTsFileName(self, tsId: str) -> str:
        pattern = f'{tsId}.mrc'
        return  self._getExtraPath(pattern) if  self._getFormValue(RE_STACK_OUT_TS) else self._getTmpPath(pattern)

    def _getArcticCmd(self, tsId: str) -> str:
        ts = self.tsDict[tsId]
        acq = ts.getAcquisition()
        cmd = [
            f'--input_ts "{ts.getFirstItem().getFileName()}"',
            f'--cleaned_ts "{self._getOutTsFileName(tsId)}"',
            f'--angle_start {acq.getAngleMin()}',
            f'--angle_step {acq.getStep()}',
            f'--model "{self._getModelFile()}"',
            f'--csv_output "{self._getTmpPath()}"'
        ]
        if self._getFormValue(GEN_PDF_REPOS):
            cmd.append(f'--pdf_output "{self._getPdfReportFn(tsId)}"')
        return ' '.join(cmd)