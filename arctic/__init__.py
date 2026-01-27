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
import os
import pwem
from arctic.constants import ARCTIC_CUDA_LIB, V_0_0_1, ARCTIC_ENV_ACTIVATION, ARCTIC_DEFAULT_ACTIVATION_CMD, ARCTIC, \
    ARCTIC_DEFAULT_VERSION, ARCTIC_GIT_REPO_URL, ARCTIC_GIT_COMMIT, ARCTIC_REPO_DIRNAME, ARTCTIC_MODELS_URL, \
    ARCTIC_ENV_NAME, MODELS_PARENT_DIR, ARCTIC_HOME

from pyworkflow import TOMO
from pyworkflow.utils import Environ

__version__ = '3.0.0'

from scipion.constants import PYTHON


# _logo = "icon.png"
# _references = ['']


class Plugin(pwem.Plugin):
    _homeVar = ARCTIC_HOME
    _pathVars = [ARCTIC_CUDA_LIB]
    _supportedVersions = [V_0_0_1]
    _url = "https://github.com/scipion-em/scipion-em-arctic"
    _processingField = [TOMO]

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(ARCTIC_HOME, f'{ARCTIC}-{ARCTIC_DEFAULT_VERSION}')
        cls._defineVar(ARCTIC_ENV_ACTIVATION, ARCTIC_DEFAULT_ACTIVATION_CMD)
        cls._defineVar(ARCTIC_CUDA_LIB, pwem.Config.CUDA_LIB)
        
    @classmethod
    def getArcticEnvActivation(cls):
        return cls.getVar(ARCTIC_ENV_ACTIVATION)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch arctic. """
        environ = Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        cudaLib = cls.getVar(ARCTIC_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

    @classmethod
    def defineBinaries(cls, env):
        ARCTIC_CLONED = f'{ARCTIC}_cloned'
        ARCTIC_CONDA_ENV_CREATED = f'{ARCTIC}_conda_env_created'
        ARCTIC_MODELS_DL = f'{ARCTIC}_models_downloaded'
        ARCTIC_INSTALLED = f'{ARCTIC}_{ARCTIC_DEFAULT_VERSION}_installed'
        MODELS_FILE = 'models.zip'

        # Clone the repository (not published in Pypi yet) and checkout to the specified commit
        articClonedRepo = cls.getHome(ARCTIC_REPO_DIRNAME)
        cloneCmd = f'[ -d {articClonedRepo} ] && rm -rf {articClonedRepo}; '  # Remove the cloned dir if exists
        cloneCmd += f'git clone {ARCTIC_GIT_REPO_URL} && '
        cloneCmd += f'cd {ARCTIC_REPO_DIRNAME} && '
        # cloneCmd += f'git checkout {ARCTIC_GIT_COMMIT} && '
        cloneCmd += f'cd .. && touch {ARCTIC_CLONED}'

        # Create the environment or update it depending on if it already exists or not
        createEnvCmd = f'{cls.getCondaActivationCmd()}'
        createEnvCmd += f'cd {ARCTIC_REPO_DIRNAME} && '
        createEnvCmd += (f" conda env list | grep -qE '^{ARCTIC_ENV_NAME}\s' && "
                         f"conda env update -n {ARCTIC_ENV_NAME} -f environment.yml || "
                         f"conda env create -n {ARCTIC_ENV_NAME} -f environment.yml && ")
        createEnvCmd += f'cd .. && touch {ARCTIC_CONDA_ENV_CREATED}'

        # Download the models
        dlModelsCmd = f'cd {ARCTIC_REPO_DIRNAME} && '
        dlModelsCmd += f'wget -O {MODELS_FILE} {ARTCTIC_MODELS_URL} && '
        dlModelsCmd += f'unzip {MODELS_FILE} && '
        dlModelsCmd += f'rm {MODELS_FILE} && '
        dlModelsCmd += f'mv {ARCTIC_REPO_DIRNAME} {MODELS_PARENT_DIR} && '  # Models dir were unzipped as ARCTiC
        dlModelsCmd+= f'cd .. && touch {ARCTIC_MODELS_DL}'

        # Flag installation finished
        installationCmd = [
            (cloneCmd, ARCTIC_CLONED),
            (createEnvCmd, ARCTIC_CONDA_ENV_CREATED),
            (dlModelsCmd, ARCTIC_MODELS_DL),
            (f'touch {ARCTIC_INSTALLED}', ARCTIC_INSTALLED)
        ]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(ARCTIC,
                       version=ARCTIC_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=installationCmd,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=True)

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = ['unzip']
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def runArctic(cls, protocol, program, args, cwd=None, numberOfMpi=1):
        """ Run arctic command from a given protocol. """
        cmd = cls.getCondaActivationCmd() + " "
        cmd += cls.getArcticEnvActivation()
        cmd += f" && CUDA_VISIBLE_DEVICES=%(GPU)s"
        cmd += f" && {PYTHON} {program} "
        protocol.runJob(cmd, args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=numberOfMpi)

