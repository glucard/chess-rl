import threading, time
from IPython.display import clear_output
from matplotlib import pyplot as plt

class PromptQValues:
    def __init__(self):
        self.q_values = []
        self.thread = threading.Thread(target=self.prompt_qvalues, daemon=True)
        self.thread.start()

    def set_qvalues(self, q_values):
        self.q_values = q_values

    def prompt_qvalues(self):
        while True:
            names = ["right", "left", "jump", "run"]
            # Figure Size
            # fig = plt.figure(figsize =(10, 7))
            # Horizontal Bar Plot
            if len(self.q_values) > 0:
                clear_output(wait=True)
                plt.bar(names[0:2], self.q_values)
                plt.ylim(-1, 1)
                self.q_values = []
                # Show Plot
                plt.show()
            time.sleep(0.3)

