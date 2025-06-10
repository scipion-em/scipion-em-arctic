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
    ARCTIC_DEFAULT_VERSION, ARCTIC_GIT_REPO, ARCTIC_GIT_COMMIT
from pyworkflow.utils import Environ

__version__ = '3.0.0'
# _logo = "icon.png"
# _references = ['']


class Plugin(pwem.Plugin):
    _pathVars = [ARCTIC_CUDA_LIB]
    _supportedVersions = [V_0_0_1]
    _url = "https://github.com/scipion-em/scipion-em-arctic"

    @classmethod
    def _defineVariables(cls):
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
        ARCTIC_INSTALLED = '%s_%s_installed' % (ARCTIC, ARCTIC_DEFAULT_VERSION)
        installationCmd = cls.getCondaActivationCmd()
        # Clone the repository (not published in Pypi yet)
        installationCmd += f'git clone {ARCTIC_GIT_REPO} && '

        # Checkout to the specified commit
        installationCmd += f'git checkout {ARCTIC_GIT_COMMIT}'

        # Create the environment
        installationCmd += ' conda env create -f environment.yml && '

        # Flag installation finished
        installationCmd += 'touch %s' % ARCTIC_INSTALLED

        ARCTIC_commands = [(installationCmd, ARCTIC_INSTALLED)]
        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(ARCTIC,
                       version=ARCTIC_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=ARCTIC_commands,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=True)

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def runArctic(cls, protocol, args, cwd=None, numberOfMpi=1):
        """ Run arctic command from a given protocol. """
        cmd = cls.getCondaActivationCmd() + " "
        cmd += cls.getArcticEnvActivation()
        cmd += f" && CUDA_VISIBLE_DEVICES=%(GPU)s {ARCTIC} "
        protocol.runJob(cmd, args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=numberOfMpi)

