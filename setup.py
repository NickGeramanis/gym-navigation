from setuptools import setup

setup(name='gym_navigation',
      version='0.1.0',
      packages=['gym_navigation',
                'gym_navigation.enums',
                'gym_navigation.envs',
                'gym_navigation.geometry'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Navigation Environment for OpenAI Gym',
      url='https://github.com/NickGeramanis/gym-navigation',
      license='GPLV3',
      python_requires='==3.10.10',
      install_requires=['gymnasium==0.28.1', 'numpy==1.24.2', 'pygame==2.3.0'])
