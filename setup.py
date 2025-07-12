from setuptools import setup

setup(name='gym_navigation',
      version='1.0.0',
      packages=['gym_navigation',
                'gym_navigation.enums',
                'gym_navigation.envs',
                'gym_navigation.geometry'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Navigation Environment for OpenAI Gym',
      url='https://github.com/NickGeramanis/gym-navigation',
      license='GPLV3',
      python_requires='==3.13.5',
      install_requires=['gymnasium==1.2.0', 'numpy==2.3.1', 'pygame==2.6.1'])
