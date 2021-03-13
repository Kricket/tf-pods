from IPython.display import display
from ipywidgets import widgets


class Log:
    def log(self, msg: str):
        print(msg)
    def replace(self, msg):
        self.log(msg)
    def reset(self):
        pass

class JupyterLog(Log):
    def __init__(self):
        self.widget = None
        self.reset()

    def reset(self):
        self.widget = widgets.Label()
        display(self.widget)

    def log(self, msg: str):
        self.widget.value += "\n" + msg

    def replace(self, msg):
        self.widget.value = msg
