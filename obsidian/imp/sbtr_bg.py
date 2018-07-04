'''
First Obsidian class: remove background features from a powder diffraction image
.. automodule:: sbtr_bg
.. moduleauthor:: Fiona Young
'''

class sbtr_bg():
	'''
	.. autoclass:: sbtr_bg
	'''
	def __init__(self, bgfile):
		'''
		:param bgfile: file of background image to be subtracted
		'''
		self.bgfile = bgfile
	
	def subtract(self, filelist):
		'''
		:param filelist: collection of files to be modified
		:returns: newfilelist modidied files
		'''	
		pass


