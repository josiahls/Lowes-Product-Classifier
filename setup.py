import re
import os
import setuptools

# Get the setuptools package imported
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


# Allow parsing form the current directory
def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


# Read the README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the version number
version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                    read('lowes-product-classifier/__init__.py'), re.MULTILINE).group(1)

# Set the final setup versions
setuptools.setup(name='lowes_product_classifier',
                 version=version,
                 python_requires=">=3.6.0, <3.7",
                 description='What the module does',
                 url='https://github.com/josiahls/lowes-product-classifier',
                 author='Josiah Laivins',
                 author_email='jlaivins@uncc.edu',
                 license='',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy', 'sklearn', 'tensorflow', 'tensor_probability', 'pillow'])
