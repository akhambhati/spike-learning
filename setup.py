from setuptools import find_packages, setup

requirements = [*map(str.strip, open("requirements.txt").readlines())]

setup(
    name='spike_learning',
    version='0.1.0',
    author='Ankit N. Khambhati',
    author_email='akhambhati@gmail.com',
    packages=find_packages(exclude=('test', 'docs')),
    include_package_data=True,
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=requirements,
    dependency_links=[
        'git+http://github.com/akhambhati/pyEisen.git@8ffc18cf3d1413c960336b48abcd3771c42c014d#egg=pyEisen-1.0.0']
)
