"""
Desc:   This file contains all the code related to errors handling
"""


class Error(object):
	"""
	The Error class. when creating an Error object, the code will exit,
	and informative prints will be shown
	"""
	def __init__(self, error, path):
		self.message = '---------- || Please notice! || ----------\n' + 'An error was caught:'
		self.error = error
		self.path = path

		self.print_error()
		self.exit_code()

	def print_error(self):
		print(self.message)
		print('Error:   ' + self.error, flush=True)
		print('Path:    ' + self.path, flush=True)

	@staticmethod
	def exit_code():
		exit()
