from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='gym_navigation',
      version='1.0.0',
      author='Nick Geramanis',
      author_email='nickgeramanis@gmail.com',
      description='Navigation Environment for OpenAI Gym',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/NickGeramanis/gym-navigation',
      project_urls={
          "Bug Tracker": "https://github.com/NickGeramanis/gym-navigation"
                         "/issues"
      },
      license='GPLV3',
      install_requires=['gym', 'numpy', 'matplotlib']
      )
