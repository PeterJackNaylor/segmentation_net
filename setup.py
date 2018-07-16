import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

exec(open('segmentation_net/version.py').read()) # loads __version__

setup(name='segmentation_net',
      version=__version__,
      author="Peter Naylor",
      author_email="peter.jack.naylor@gmail.com",
      description="Useful code for segmentation networks in tensorflow",
      long_description=open('README.rst').read(),
      long_description_content_type="text/markdown",
      license='see LICENSE.txt',
      url="https://github.com/PeterJackNaylor/segmentation_net",
      packages=find_packages(exclude='docs'),
      classifiers=(
          "Programming Language :: Python :: 2",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ),
)
