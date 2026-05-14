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
from os.path import join, exists
from typing import Union, Dict
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
    """
    Performs automated detection and removal of corrupted tilt-images in cryo-electron tomography tilt-series.
    The protocol identifies acquisition artifacts such as drift, ice reflections, lamella edges, thick lamella
    regions, and contamination in order to improve the overall quality of tomographic datasets before downstream
    reconstruction and analysis. Its main purpose is to assist users in cleaning tilt-series in a consistent,
    reproducible, and scalable manner using deep learning classification models trained specifically for Cryo-ET
    data quality assessment. More info:
    https://www.biorxiv.org/content/10.1101/2025.03.13.642992v1

    AI Generated:

    Automated Removal of Corrupted Tilts (ProtArcticRemoveCorruptedTilts) - User Manual
        Overview

        The Automated Removal of Corrupted Tilts protocol is designed to evaluate Cryo-ET tilt-series and detect
        tilt-images that are likely to compromise tomographic reconstruction quality. In practical cryo-electron
        tomography workflows, tilt-series frequently contain images affected by acquisition instabilities,
        contamination, excessive specimen thickness, charging effects, or geometric artifacts near the lamella edge.
        These problematic views can strongly reduce reconstruction quality, introduce alignment instability, and
        obscure biologically meaningful structural information.

        The protocol applies pretrained machine learning models to classify each tilt-image as acceptable or
        corrupted. The resulting cleaned tilt-series can then be used in downstream processing steps such as tilt
        alignment, tomographic reconstruction, subtomogram averaging, or segmentation. By automating this quality
        control stage, the protocol reduces the amount of manual inspection required while promoting reproducibility
        across large datasets.

        Biological Motivation and Use Cases

        In Cryo-ET experiments, the quality of individual tilt-images can vary significantly across the angular
        range. High tilt angles are especially vulnerable to low signal-to-noise ratio, ice contamination,
        specimen thickening, and beam-induced artifacts. Even a relatively small number of severely corrupted
        views may negatively affect the final tomogram and complicate interpretation of macromolecular structures
        or cellular environments.

        This protocol is particularly useful in large-scale tomography facilities, high-throughput acquisition
        campaigns, or screening projects where manual inspection becomes impractical. It is also beneficial when
        preparing datasets for sensitive downstream analyses such as subtomogram classification, in situ structural
        biology studies, or quantitative measurements that depend on reconstruction fidelity.

        Input Tilt-Series and Initial Considerations

        The protocol requires a set of tilt-series as input. Existing excluded views present in the input metadata
        are preserved and respected during processing. Users should ensure that the tilt-series acquisition
        parameters are properly defined because the angular information is important for interpreting the dataset
        consistently across the full tilt range.

        Before running the protocol, it is advisable to verify that the tilt-series are already reasonably aligned
        and organized. The protocol focuses on image quality assessment rather than correcting geometric alignment
        problems or acquisition metadata inconsistencies.

        Deep Learning Models and Classification Strategy

        Several pretrained models are available for image classification, each providing different balances between
        computational efficiency and detection sensitivity. Transformer-based architectures generally provide strong
        overall performance and are particularly effective in distinguishing clean images from corrupted ones. These
        models are often preferred for demanding biological datasets where preserving high-quality information is
        critical.

        Lightweight transformer variants offer faster execution and lower computational requirements, making them
        suitable for exploratory analyses or systems with limited GPU resources. Convolutional neural network models
        provide robust corrupted-image detection and may perform well in workflows prioritizing conservative
        filtering strategies.

        From a biological perspective, users should understand that different models may emphasize sensitivity or
        specificity differently. Conservative models may preserve borderline images that still contain useful signal,
        whereas aggressive models may remove more questionable views at the risk of discarding partially informative
        data.

        Handling of Corrupted Tilt-Images

        The protocol supports two principal strategies for handling corrupted images. In metadata-only mode, the
        original tilt-series binary files remain unchanged while problematic views are marked as excluded in the
        metadata. This approach preserves the original acquisition files and is generally recommended when users
        want maximum traceability and compatibility with downstream software that supports excluded-view handling.

        Alternatively, the protocol can generate newly stacked tilt-series containing only the accepted images.
        This option physically removes corrupted views from the output binary data and may simplify downstream
        workflows that expect already cleaned stacks. However, users should recognize that re-stacking modifies the
        original image organization and may complicate later comparisons with the raw acquisition files.

        Threshold for Acceptable Tilt-Series

        A biologically important parameter is the maximum tolerated percentage of corrupted tilt-images within a
        tilt-series. This threshold determines whether a tilt-series is considered acceptable after cleaning or
        whether it should instead be flagged as globally problematic.

        For example, a tilt-series containing a small number of excluded images may still produce a reliable
        tomographic reconstruction, especially if the missing views are distributed across the angular range.
        Conversely, a dataset with excessive corruption may suffer from strong missing-angle artifacts, alignment
        instability, or severe reconstruction degradation. The protocol therefore separates severely compromised
        tilt-series into a dedicated output category so they can be reviewed independently.

        PDF Reports and Dataset Inspection

        The protocol can optionally generate PDF reports summarizing the classification results for each tilt-series.
        These reports provide a convenient way to visually inspect the distribution of excluded images and evaluate
        the confidence of the classification results across the angular range.

        In practical biological workflows, these reports are extremely valuable for quality control because they
        allow users to determine whether the excluded images correspond to expected problematic regions, such as
        high tilts or contamination events. They also help identify systematic acquisition issues that may affect
        multiple datasets collected during the same microscope session.

        GPU Usage and Performance Considerations

        Because the protocol relies on deep learning inference, GPU acceleration is strongly recommended for large
        datasets. High-throughput tomography projects containing hundreds of tilt-series can benefit substantially
        from GPU execution, particularly when using transformer-based models.

        Users working on smaller exploratory datasets or computationally limited systems may prefer lightweight
        models that reduce execution time while still providing useful filtering performance. In general, balancing
        model complexity with dataset size and biological requirements leads to the most efficient workflows.

        Outputs and Biological Interpretation

        The protocol produces cleaned tilt-series together with a separate collection of tilt-series considered too
        corrupted for reliable downstream use. Acceptable tilt-series preserve the acquisition structure while
        excluding problematic images according to the selected strategy.

        From a biological interpretation standpoint, users should remember that removing excessive numbers of views
        can reduce angular coverage and increase reconstruction anisotropy. Therefore, aggressive cleaning should
        always be balanced against the need to preserve sufficient angular information for accurate tomographic
        reconstruction.

        Practical Recommendations

        For routine Cryo-ET processing, it is often advisable to begin with a balanced classification model and a
        moderate tolerance threshold. The resulting outputs should then be inspected visually, especially during
        early adoption of the protocol or when working with unfamiliar sample types.

        If many biologically meaningful images appear to be removed, users may consider increasing the allowed
        percentage threshold or selecting a less aggressive model. Conversely, if reconstructed tomograms still
        exhibit strong artifacts, stricter filtering may improve overall quality.

        In high-quality datasets with relatively homogeneous imaging conditions, metadata-only exclusion is often
        sufficient. For highly automated downstream pipelines or external software environments, generating cleaned
        re-stacked tilt-series may simplify integration.

        Final Perspective

        Automated corruption detection in Cryo-ET represents an important step toward scalable and reproducible
        tomography workflows. By systematically identifying problematic tilt-images, this protocol helps improve
        reconstruction quality, reduces manual inspection burden, and supports more reliable downstream biological
        interpretation. Careful selection of classification models, filtering thresholds, and output strategies is
        essential for balancing data preservation with reconstruction fidelity.
    """

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
        try:
            ts = self.tsDict[tsId]
            tsTmpFile = self.getTsTmpFile(tsId)
            if ts.hasExcludedViews():
                logger.info(cyanStr(f'tsId = {tsId} -> Excluded views detected in the input tilt-series. Re-stacking '
                                    f'without them...'))
                ts.applyTransform(tsTmpFile)
            else:
                # This way the input file when calling the program will be located in tmp, allowing extra checking
                # regarding if a file was re-stacked or not
                createLink(ts.getFirstItem().getFileName(), tsTmpFile)
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> input conversion failed with the exception -> {e}'))


    def runArctic(self, tsId: str):
        if tsId in self.failedTsIds:
            return
        try:
            logger.info(cyanStr(f'tsId = {tsId} -> Running ARCTiC...'))
            Plugin.runArctic(self, self._getProgram(), self._getArcticCmd(tsId))
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} - Failed to process with exception {e}'))

    def createOutputStep(self, tsId: str):
        if tsId in self.failedTsIds:
            return
        try:
            with self._lock:
                # Read the corresponding csv file
                resDict = self._readCsvreport(tsId)
                if resDict:
                    self._createOutput(tsId, resDict)
                else:
                    logger.error(redStr(f'tsId = {tsId} -> skipping...'))
        except Exception as e:
            logger.error(redStr(f'tsId = {tsId} -> Unable to register the output with exception '
                                f'{e}. Skipping...'))

    def closeOutputSetStep(self):
        outBadTsSet = getattr(self, ArcticOutputs.badTiltSeries.name, None)
        outTsSet = getattr(self, ArcticOutputs.tiltSeries.name, None)
        if not outTsSet and not outBadTsSet:
            raise Exception('No outputs were generated by ARCTiC')
        else:
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

    def _readCsvreport(self, tsId: str) -> Union[Dict[int, bool] , None]:
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
        csvFileName = self._getCsvReportFn(tsId)
        if not exists(csvFileName):
            logger.error(redStr(f'tsId = {tsId} -> no csv report was found: {csvFileName}'))
            return None
        with open(csvFileName, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            resDict = {int(row[CURRENT_INDEX]): row[TO_BE_REMOVED] == "True" for row in reader}
        return resDict

    def _getOutTsFileName(self, tsId: str) -> str:
        pattern = f'{tsId}.mrc'
        return  self._getExtraPath(pattern) if  self._getFormValue(RE_STACK_OUT_TS) else self._getTmpPath(pattern)

    def _getArcticCmd(self, tsId: str) -> str:
        ts = self.tsDict[tsId]
        acq = ts.getAcquisition()
        cmd = [
            # f'--input_ts "{ts.getFirstItem().getFileName()}"',
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

    def _createOutput(self, tsId: str, resDict: Dict[int, bool]):
        ts = self.tsDict[tsId]
        indices = resDict.keys()
        toBeRemovedList = resDict.values()
        # Check the percentage of bad tilt-images
        nImgs = len(indices)
        nBadTi = sum(toBeRemovedList)
        allowedNBadTilts = round(0.01 * self._getFormValue(NO_GO_PERCENT) * nImgs)
        # Set of tilt-series
        outTsSet = self._getOutTsSet(attrName=ArcticOutputs.badTiltSeries.name)
        outTs = TiltSeries()
        outTs.copyInfo(ts)
        outTsSet.append(outTs)
        # Tilt-series
        if nBadTi > allowedNBadTilts:
            logger.info(cyanStr(f'tsId = {tsId} -> too many bad tilt-images detected [{nBadTi} > '
                                f'{allowedNBadTilts}]. Stored as bad tilt-series.'))
            for ti in ts.iterItems():
                outTi = TiltImage()
                outTi.copyInfo(ti)
                outTs.append(outTi)
        else:
            logger.info(cyanStr(f'tsId = {tsId} -> bad tilt-images detected [{nBadTi} <= '
                                f'{allowedNBadTilts}]. Stored as good tilt-series.'))
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
                    if ti.getAcquisitionOrder() in presentAcqOrders:
                        goodTi = not resDict[ind]
                        outTi.setEnabled(goodTi)
                    outTs.append(outTi)
        # Register the data of the current ts and update the tsSet
        outTs.write()
        outTsSet.update(outTs)
        outTsSet.write()
        self._store(outTsSet)
