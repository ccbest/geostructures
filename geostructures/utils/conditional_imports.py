"""
Intercepts import errors concerning optional imports to either:
    - Provide a more detailed error response or
    - Auto-download the specified package
"""

__all__ = ['ConditionalPackageInterceptor']

from importlib import util
import subprocess
import sys
from typing import Union

from geostructures.utils.mixins import LoggingMixin


class ConditionalPackageInterceptor(LoggingMixin):
    """
    Provides automatic pip installation of a package if it isn't found. Only packages added to this
    object using the .permit_packages() method are allowed to be automatically installed.

    To use:
        In your code's entrypoint, add the following code:

            PipInstaller.permit_packages(
                <list or dict of packages>
            )
            sys.meta_path.append(PipInstaller)

    Use Case:
        If you're building a library where only some of the functionality relies on third-party
        packages, and you anticipate that only some users will ever use that functionality, then
        you may want to only install those packages if they're actually needed.

    Note that this object needs to be defined before all of your other code runs or else it won't
    work. In most cases, the closest-to-root __init__.py is a great place.

    """

    PERMITTED_PACKAGES: dict = {}
    AUTO_DOWNLOAD = False

    @classmethod
    def permit_packages(cls, packages: Union[list, dict]) -> None:
        """
        Adds python packages to the list of packages that are permitted to be automatically
        installed.

        The name of a python package is not always the same between "pip install <package>" and
        "import <package>". Blindly installing any package that isn't found can therefore be not
        just a programmatic risk but a security one as well (malicious packages do exist).

        You can add to this list in two ways:
            As a list: packages will be pip installed exactly as listed
                ["xlrd"]
                "import xlrd" -> pip install xlrd

            As a dict: packages will be pip installed by the corresponding key
                {"yaml": "pyyaml"}
                "import yaml" -> pip install pyyaml

        Args:
            packages (Union[list, dict]): The packages that will be allowed to auto-install if
                                          missing

        Returns:
            None
        """
        if isinstance(packages, list):
            cls.PERMITTED_PACKAGES.update({item: item for item in packages})
        elif isinstance(packages, dict):
            cls.PERMITTED_PACKAGES.update(packages)
        else:
            raise TypeError(
                f"Permitted packages must be submitted as a list or dict, not {type(packages)}"
            )

    @classmethod
    def permit_auto_download(cls, option: bool) -> None:
        """
        Defines whether packages may be auto-downloaded or not. Default False.
        """
        cls.AUTO_DOWNLOAD = option

    @classmethod
    def find_spec(  # pylint: disable=unused-argument, inconsistent-return-statements
            cls, name, path, target=None
    ):
        """
        DO NOT USE.

        Will be executed by python's importlib when it's search for installed packages. This object
        will be examined last since you .append()'ed it to the sys.meta_path (you did follow the
        instructions in the docstring, right?), which means that if importlib got this far without
        finding the package then the package doesn't exist in your virtual environment.

        Will only pip install if the package name is in the PERMITTED_PACKAGES class variable.
        Otherwise you'll get the usual "ModuleNotFoundError".

        Args:
            name (str): The name of the package
            path:
            target:

        Returns:

        """
        if name not in cls.PERMITTED_PACKAGES:
            return

        if cls.AUTO_DOWNLOAD:
            print(f"Module {name!r} not installed. Attempting to pip install...")
            try:
                subprocess.run(
                    f"{sys.executable} -m pip install {cls.PERMITTED_PACKAGES[name]}",
                    check=True
                )
            except subprocess.CalledProcessError:
                return None

            return util.find_spec(name)

        raise ModuleNotFoundError(
            f"You are attempting to use a module which requires an optional installation ({name}). "
            "Please choose one of the following options to continue: \n\n "
            "1) Enable package auto-installation using: \n"
            "    # In your root-most __init__.py file:"
            "    from skopeutils.package_interceptor import ConditionalPackageInterceptor \n"
            "    ConditionalPackageInterceptor.permit_auto_download(True) \n\n"
            "2) Pip install the package yourself using the following command: \n"
            f"    pip install {cls.PERMITTED_PACKAGES[name]}"
        )
