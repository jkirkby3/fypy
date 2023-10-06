from setuptools import setup, find_packages

setup(
  name='fypy',
  version='0.0.1',
  description='Financial Quant model calibration and option pricing',
  long_description='Financial Quant model calibration and option pricing',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering ',
    "Operating System :: OS Independent",
  ],
  keywords='quant option-pricing calibration fit levy stochastic-volatility',
  url='https://github.com/jkirkby3/fypy',
  author='Justin Lars Kirkby',
  author_email='jkirkby33@gmail.com',
  license='MIT',
  packages=find_packages(),
  python_requires=">=3.7",
  install_requires=[
    "matplotlib",
    "numpy",
    "python-dateutil",
    "scipy",
    "py_lets_be_rational>=1.0.1",
    "pandas",
    "requests-cache",
    "yfinance",
  ],
  extras_require={
    "test": []
  },
  include_package_data=True,
  zip_safe=False,
)
