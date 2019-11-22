## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['persondetection','connected_components','utils','yolov3','yolov3.utils'],
    package_dir={'': 'include'},
)
setup(**setup_args)
