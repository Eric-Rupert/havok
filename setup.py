from setuptools import setup

setup(
	name = "havok",
	version = "0.0.1",
	author = "Eric Rupert",
	packages = ['havok'],
	install_requires = ['numpy','scipy','sklearn.preprocessing','control.matlab','os','matplotlib.pyplot','mpl_toolkits.mplot3d']
)
