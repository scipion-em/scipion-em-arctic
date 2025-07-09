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
import csv
import logging
from collections import OrderedDict
from enum import Enum
from os.path import join
from typing import Union
from arctic import Plugin, ARCTIC_REPO_DIRNAME, MODELS_PARENT_DIR
from arctic.constants import BINARY_MODELS_DIR
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer, Set
from pyworkflow.protocol import PointerParam, BooleanParam, LEVEL_ADVANCED, EnumParam, STEPS_PARALLEL, GPU_LIST, \
    StringParam, IntParam, GE, LE
from pyworkflow.utils import Message, makePath, redStr, cyanStr, createLink
from tomo.objects import SetOfTiltSeries, TiltImage, TiltSeries

logger = logging.getLogger(__name__)

# Form variables
IN_TS_SET = 'inTsSet'
RE_STACK_OUT_TS = 'reStackOutTsSet'
NO_GO_PERCENT = 'noGoPercent'
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

# CSV report column names
CURRENT_INDEX = 'CurrentIndex'
TO_BE_REMOVED = 'ToBeRemoved'
REMOVED = 'Removed'

# Other vars
PDF_REPORT_DIR = 'reports'


class ArcticOutputs(Enum):
    tiltSeries = SetOfTiltSeries()
    badTiltSeries = SetOfTiltSeries()


class ProtArcticRemoveCorruptedTilts(EMProtocol):
    """Automated Removal of Corrupted Tilts in Cryo-ET. Corrupted tilt-images are the ones
    in which any of these effects/artifacts is detected: drift, ice reflection, lamella edge
    thick lamella, and contamination. More details in
    https://www.biorxiv.org/content/10.1101/2025.03.13.642992v1."""

    _label = 'automated removal of corrupted tilts'
    _devStatus = BETA
    _possibleOutputs = ArcticOutputs
    stepsExecutionMode = STEPS_PARALLEL
    program = 'run_TS_cleaning.py'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tsDict = None
        self.failedTsIds = []

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

        form.addParam(NO_GO_PERCENT, IntParam,
                      default=50,
                      label='Allowed percentage of bad tilt-images',
                      validators=[GE(0), LE(100)],
                      help="Percentage, in range [0, 100], of bad tilt-images, per tilt-series, allowed to consider "
                           "a tilt-series to be acceptable. "
                           "\n\nFor example, if a tilt-series has 40 tilt-images, the "
                           "percentage is 50 and the number of tilt-images excluded by the program is greater than "
                           "20 (which is the 50% of 40), the corresponding tilt-series will be part of the output "
                           "called 'badTiltSeries' without any change."
                           "\n\nOn the other side, if the number of tilt-images excluded by the program is lower or "
                           "equal to 20, the corresponding tilt-series will appear in the output called 'tiltSeries', "
                           "considering the results of the program.")

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
            pid1 = self._insertFunctionStep(self.convertInputStep, tsId,
                                             prerequisites=[],
                                             needsGPU=False)
            pid2 = self._insertFunctionStep(self.runArctic, tsId,
                                             prerequisites=pid1,
                                             needsGPU=True)
            pid3 = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=pid2,
                                              needsGPU=False)
            closeSetStepDeps.append(pid3)

        self._insertFunctionStep(self.closeOutputSetStep,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.tsDict = {ts.getTsId(): ts.clone() for ts in self._getTsSet()}
        if self._getFormValue(GEN_PDF_REPOS):
            makePath(self._getPdfReportDir())


    def convertInputStep(self, tsId: str):
        ts = self.tsDict[tsId]
        presentAcqOrders = ts.getTsPresentAcqOrders()
        tsTmpFile = self.getTsTmpFile(tsId)
        if presentAcqOrders:
            logger.info(cyanStr(f'tsId = {tsId} -> Excluded views detected in the input tilt-series. Re-stacking '
                                f'without them...'))
            ts.applyTransform(tsTmpFile, presentAcqOrders=presentAcqOrders)
        else:
            # This way the input file when calling the program will be located in tmp, allowing extra checking
            # regarding if a file was re-stacked or not
            createLink(ts.getFirstItem().getFileName(), tsTmpFile)

    def runArctic(self, tsId: str):
        try:
            logger.info(cyanStr(f'tsId = {tsId} -> Running ARCTiC...'))
            Plugin.runArctic(self, self._getProgram(), self._getArcticCmd(tsId))
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} - Failed to process with exception {e}'))

    def createOutputStep(self, tsId: str):
        # TODO: consider the possible pre-excluded views here
        if tsId not in self.failedTsIds:
            ts = self.tsDict[tsId]
            # Read the corresponding csv file
            resDict = self._readCsvreport(tsId)
            if resDict:
                indices = resDict.keys()
                toBeRemovedList = resDict.values()
                # Check the percentage of bad tilt-images
                nImgs = len(indices)
                nBadTi = sum(toBeRemovedList)
                allowedNBadTilts = round(0.01 * self._getFormValue(NO_GO_PERCENT) * nImgs)
                if nBadTi > allowedNBadTilts:
                    logger.info(cyanStr(f'tsId = {tsId} -> too many bad tilt-images detected [{nBadTi} > '
                                        f'{allowedNBadTilts}]. Stored as bad tilt-series.'))
                    outTsSet = self._getOutTsSet(attrName=ArcticOutputs.badTiltSeries.name)
                    outTs = TiltSeries()
                    outTs.copyInfo(ts)
                    outTsSet.append(outTs)
                    for ti in ts.iterItems():
                        outTi = TiltImage()
                        outTi.copyInfo(ti)
                        outTs.append(outTi)
                    # Register the data of the current ts and update the tsSet
                    outTs.write()
                    outTsSet.update(outTs)
                    outTsSet.write()
                    self._store(outTsSet)

                else:
                    logger.info(cyanStr(f'tsId = {tsId} -> bad tilt-images detected [{nBadTi} <= '
                                        f'{allowedNBadTilts}]. Stored as good tilt-series.'))
                    outTsSet = self._getOutTsSet(attrName=ArcticOutputs.tiltSeries.name)
                    outTs = TiltSeries()
                    outTs.copyInfo(ts)
                    outTsSet.append(outTs)
                    if self._getFormValue(RE_STACK_OUT_TS):
                        outTsFileName = self._getOutTsFileName(tsId)
                        for ind, ti in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
                            goodTi = not resDict[ind]
                            if goodTi:
                                outTi = TiltImage()
                                outTi.copyInfo(ti)
                                outTi.setFileName(outTsFileName)
                                outTs.append(ti)
                    else:
                        outTsFileName = ts.getFirstItem().getFileName()
                        presentAcqOrders = ts.getTsPresentAcqOrders()
                        for ind, ti in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
                            outTi = TiltImage(tsId)
                            outTi.copyInfo(ti)
                            outTi.setFileName(outTsFileName)
                            # Only update the status of the non-excluded views from the input ts,
                            # which are the ones that have been processed by arctic
                            if ti.getAcqOrder() in presentAcqOrders:
                                goodTi = not resDict[ind]
                                outTi.setEnabled(goodTi)
                            outTs.append(outTi)
                    # Register the data of the current ts and update the tsSet
                    outTs.write()
                    outTsSet.update(outTs)
                    outTsSet.write()
                    self._store(outTsSet)

    def closeOutputSetStep(self):
        outBadTsSet = getattr(self, ArcticOutputs.badTiltSeries.name, None)
        outTsSet = getattr(self, ArcticOutputs.tiltSeries.name, None)
        if not (outTsSet and outBadTsSet):
            raise Exception('No outputs were generated by ARCTiC')
        super()._closeOutputSet()

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

    def getTsTmpFile(self, tsId: str) -> str:
        return self._getTmpPath(f'{tsId}.mrc')

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

    def _getCsvReportFn(self, tsId: str) -> str:
        return self._getTmpPath(f'{tsId}.csv')

    def _readCsvreport(self, tsId: str) -> Union[dict, None]:
        """Reads the corresponding csv file. It looks like this:
        CurrentIndex,ToBeRemoved,Removed
        0,True,False
        1,False,False
        2,False,False
        3,True,False
        4,False,False
        [...]

        :return resDict: dictionary of keys = indices and values = ToBeRemoved flag.
        """
        resDict = None
        csvFileName = self._getCsvReportFn(tsId)
        with open(csvFileName, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            resDict = {int(row[CURRENT_INDEX]): row[TO_BE_REMOVED] == "True" for row in reader}
        if not resDict:
            logger.error(redStr(f'tsId = {tsId} -> no csv report was found: {csvFileName}'))
        return resDict

    def _getOutTsFileName(self, tsId: str) -> str:
        pattern = f'{tsId}.mrc'
        return  self._getExtraPath(pattern) if  self._getFormValue(RE_STACK_OUT_TS) else self._getTmpPath(pattern)

    def _getArcticCmd(self, tsId: str) -> str:
        ts = self.tsDict[tsId]
        acq = ts.getAcquisition()
        cmd = [
            f'--input_ts "{self.getTsTmpFile(tsId)}"',
            f'--cleaned_ts "{self._getOutTsFileName(tsId)}"',
            f'--angle_start {acq.getAngleMin()}',
            f'--angle_step {acq.getStep()}',
            f'--model "{self._getModelFile()}"',
            f'--csv_output "{self._getCsvReportFn(tsId)}"'
        ]
        if self._getFormValue(GEN_PDF_REPOS):
            cmd.append(f'--pdf_output "{self._getPdfReportFn(tsId)}"')
        return ' '.join(cmd)

    def _getOutTsSet(self, attrName: str) -> SetOfTiltSeries:
        inTsSetPointer = self._getTsSet(returnPointer=True)
        inTsSet = inTsSetPointer.get()
        suffix = '' if attrName == ArcticOutputs.tiltSeries.name else 'bad'
        outTsSet = getattr(self, attrName, None)
        if outTsSet:
            outTsSet.enableAppend()
        else:
            outTsSet = SetOfTiltSeries.create(self._getPath(), template='tiltseries', suffix=suffix)
            outTsSet.copyInfo(inTsSet)
            outTsSet.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{attrName: outTsSet})
            self._defineSourceRelation(inTsSetPointer, outTsSet)
        return outTsSet
