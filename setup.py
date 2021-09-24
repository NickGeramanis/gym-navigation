from setuptools import setup

setup(name='gym_navigation',
      version='0.1.0',
      packages=['gym_navigation',
                'gym_navigation.envs',
                'gym_navigation.utils'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Navigation Environment for OpenAI Gym',
      url='https://github.com/NickGeramanis/gym-navigation',
      license='GPLV3',
      install_requires=['gym', 'numpy', 'matplotlib'])
