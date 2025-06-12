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

ARCTIC = 'arctic'
ARCTIC_HOME = 'ARCTIC_HOME'
ARCTIC_REPO_DIRNAME = 'ARCTiC'
ARCTIC_GIT_REPO_URL = 'https://github.com/turonova/ARCTiC.git'
ARCTIC_GIT_COMMIT = '9d8e22df79329c66bcd489b4d65a7e993a3ab3a4'  # SHA of commit on May 27, 2025
ARTCTIC_MODELS_URL = 'https://oc.biophys.mpg.de/owncloud/s/zmMZPr2TEB4Bwda/download'
MODELS_PARENT_DIR = 'models'
BINARY_MODELS_DIR = 'binary_models'

# Supported versions
V_0_0_1 = '0.0.1'
ARCTIC_DEFAULT_VERSION = V_0_0_1

ARCTIC_ENV_NAME = 'arctic'
ARCTIC_ENV_ACTIVATION = 'ARCTIC_ENV_ACTIVATION'
ARCTIC_DEFAULT_ACTIVATION_CMD = 'conda activate %s' % ARCTIC_ENV_NAME
ARCTIC_CUDA_LIB = 'ARCTIC_CUDA_LIB'