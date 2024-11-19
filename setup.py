from setuptools import setup

setup(
    name='tangled-adjudicate',
    version='0.0.1',
    packages=['tests', 'tangled_adjudicate', 'tangled_adjudicate.utils', 'tangled_adjudicate.schrodinger', 'tangled_adjudicate.adjudicators'],
    url='https://www.snowdropquantum.com/',
    license='MIT',
    author='Geordie Rose',
    author_email='geordie@snowdropquantum.com',
    description='Tangled adjudicators',
    install_requires=['dwave-ocean-sdk', 'matplotlib']
)
