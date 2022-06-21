import setuptools

setuptools.setup(
    name='sempler',
    version='0.2.2',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['sempler', 'sempler.test'],
    scripts=[],
    url='https://github.com/juangamella/sempler',
    license='BSD 3-Clause License',
    description='Sample from general structural causal models (SCMs).',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.15.0', 'matplotlib', 'networkx>=2.2']
)
