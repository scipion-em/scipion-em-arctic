=========================
Scipion plugin for arctic
=========================

.. image:: https://img.shields.io/pypi/v/scipion-em-arctic.svg
        :target: https://pypi.python.org/pypi/scipion-em-arctic
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-arctic.svg
        :target: https://pypi.python.org/pypi/scipion-em-arctic
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-arctic.svg
        :target: https://pypi.python.org/pypi/scipion-em-arctic
        :alt: Supported Python versions

.. image:: https://img.shields.io/sonar/quality_gate/scipion-em_scipion-em-arctic?server=https%3A%2F%2Fsonarcloud.io
        :target: https://sonarcloud.io/dashboard?id=scipion-em_scipion-em-arctic
        :alt: SonarCloud quality gate

.. image:: https://img.shields.io/pypi/dm/scipion-em-arctic
        :target: https://pypi.python.org/pypi/scipion-em-arctic
        :alt: Downloads

This plugin provide a wrapper around the program `arctic <https://github.com/turonova/ARCTiC/tree/main>`_ to use it within
`Scipion <https://scipion-em.github.io/docs/release-3.0.0/index.html>`_ framework.

Installation
------------

You will need to use `3.0 <https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html>`_ 
version of Scipion to be able to run these protocols. To install the plugin, you have two options:


a) Stable version:

.. code-block::

    scipion3 installp -p scipion-em-arctic

b) Developer's version

    * download the repository from github:

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-arctic.git

    * install:

    .. code-block::

        scipion3 installp -p /path/to/scipion-em-arctic --devel

To check the installation, simply run the following Scipion test for the plugin:

    .. code-block::

        scipion3 tests arctic.tests.tests_arctic.TestArctic

Licensing
---------

arctic software package is available under `GNU General Public License v3.0 <https://opensource.org/license/gpl-3-0>`_

Protocols
---------

* **Tilt-series: automated removal of corrupted tilts.**

Latest plugin versions
----------------------

If you want to check the latest version and release history go to `CHANGES <https://github.com/scipion-em-arctic/arctic/blob/master/CHANGES.txt>`_