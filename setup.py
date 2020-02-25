from setuptools import setup, find_packages

setup(name='comet',
      version='1.0',
      description='Codebase for releasing comet model code',
      # url='http://github.com/storborg/funniest',
      author='Antoine Bosselut',
      author_email='antoineb@allenai.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        "ftfy",
        "tqdm",
        "pandas"
      ],
      zip_safe=False)
