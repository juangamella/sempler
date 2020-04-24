from distutils.core import setup

setup(
    name='sempler',
    version='0.0.1',
    author='Juan Luis Gamella',
    author_email='juangamella@gmail.com',
    packages=['sempler', 'sempler.test'],
    scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    url='http://pypi.python.org/pypi/sempler/',
    license='LICENSE.txt',
    description='Sample from arbitrary structural equation models (SEMs)',
    long_description=open('README.txt').read(),
    install_requires=[
    ],
)
