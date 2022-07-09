from setuptools import setup

setup(name='gym_navigation',
      version='0.1.0',
      packages=['gym_navigation',
                'gym_navigation.envs',
                'gym_navigation.geometry'],
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Navigation Environment for OpenAI Gym',
      url='https://github.com/NickGeramanis/gym-navigation',
      license='GPLV3',
      python_requires='>3.10.5',
      install_requires=['gym==0.24.1', 'numpy==1.23.1', 'matplotlib==3.5.2',
                        'pyqt5==5.15.7'])
