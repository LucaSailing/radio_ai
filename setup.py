from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='radio_ai_package',
      version="0.0.1",
      description="Finding fractures in X-rays",
      license="MIT",
      author="Luca, Marwan, Modibo and Mariana",
      #url="https://github.com/LucaSailing/radio_ai",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from GRAZPED.in
      include_package_data=True,
      zip_safe=False)
