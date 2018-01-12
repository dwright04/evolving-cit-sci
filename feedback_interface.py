# General imports
import os
import itertools
import numpy as np
import scipy.io as sio
from random import shuffle

# UI imports
import wx
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas

# Image processing imports
from scipy import misc
from skimage.transform import resize
from skimage.filters import threshold_otsu

# Machine learning imports
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

def get_training_example(file, plot=False):
  image = misc.imread(file)
  h, w = image.shape
  image = resize(image, (100, w)) # resize to 100 pixel height, preserving original width
  thresh = threshold_otsu(image)  # calculate threshold to cast the image as a binary
  image = np.asarray(image < thresh, dtype='int') # cast to binary image
  if plot:
    # visualise the image
    plt.imshow(image, cmap='gray_r')
    plt.axis('off')
    plt.show()
  return image

def build_model(summary=False):
  # build a model similar to above, but smaller for the time being.
  model = Sequential()
  model.add(LSTM(8, input_shape=(None, 100)))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  if summary:
    model.summary()
  return model

def get_es_guesses(image, w, npop=10, sigma=0.1, plot=False):
  # perturb the weights of the RNN and make a prediction
  images = []
  w_tries = []
  for i in range(npop):
    # clone the original model
    model_copy = Sequential()
    model_copy.add(LSTM(8, input_shape=(None, 100)))
    model_copy.add(Dense(2, activation='softmax'))
    model_copy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    w_try = w[:]
    for j in range(len(w)):
      try:
        w_try[j] += sigma * np.random.randn(w[j].shape[0], w[j].shape[1])
      except IndexError:
        w_try[j] += sigma * np.random.randn(w[j].shape[0])
    model_copy.set_weights(w_try)
    w_tries.append(w_try)
    pred = np.argmax(model_copy.predict(image.T[:,np.newaxis,:]), axis=1)
    img = np.zeros((image.shape[0], image.shape[1], 3))
    img[:,:,0] += image
    for i in range(len(pred)):
      img[:,i,1] += pred[i]
    images.append(img)
    if plot:
      plt.imshow(img)
      plt.axis('off')
      plt.show()
  return images, w_tries

def n_choose_k(n, k):
  import math
  return (math.factorial(n) / (math.factorial(n-k)*math.factorial(k)))

class mainFrame(wx.Frame):
  """ The main frame of the feedback interface
  """
  title = 'Main Console'
  def __init__(self, images):
    # 10x10 inches, 100 dots-per-inch, so set size to (1000,1000)
    wx.Frame.__init__(self, None, -1, self.title, size=(750,500), pos=(50,50))
    self.images = images
    self.npop = len(self.images)
    self.rewards = {}
    for i in range(self.npop):
      self.rewards[i] = 0
    self.comparisons = list(itertools.combinations(range(self.npop), 2))
    shuffle(self.comparisons)
    
    self.dpi = 100
    self.fig = plt.figure(figsize=(7.0, 4.5), dpi=self.dpi)
    self.fig.subplots_adjust(left=0.1, bottom=0.01, right=0.9, \
                             top=0.99, wspace=0.05, hspace=0.05)

    self.ax1 = self.fig.add_subplot(121)
    self.ax2 = self.fig.add_subplot(122)

    self.create_main_panel()

  def create_main_panel(self):
    self.panel = wx.Panel(self)
    self.draw()
    self.canvas = FigCanvas(self.panel, -1, self.fig)

    self.left_button = wx.Button(self.panel, -1, label="Left is better")
    self.left_button.Bind(wx.EVT_BUTTON, self.on_left)
    self.right_button = wx.Button(self.panel, -1, label="Right is better")
    self.right_button.Bind(wx.EVT_BUTTON, self.on_right)
    self.equal_button = wx.Button(self.panel, -1, label="Can't tell")
    self.equal_button.Bind(wx.EVT_BUTTON, self.on_equal)
    
    self.hbox = wx.BoxSizer(wx.HORIZONTAL)
    self.hbox.Add(self.left_button, 0, flag=wx.CENTER | wx.BOTTOM)
    self.hbox.Add(self.equal_button, 0, flag=wx.CENTER | wx.BOTTOM)
    self.hbox.Add(self.right_button, 0, flag=wx.CENTER | wx.BOTTOM)
    
    self.vbox = wx.BoxSizer(wx.VERTICAL)
    self.vbox.Add(self.canvas, 0, flag=wx.CENTER | wx.CENTER | wx.GROW)
    self.vbox.Add(self.hbox, border=20, flag=wx.CENTER | wx.ALIGN_CENTER_VERTICAL)
    self.panel.SetSizer(self.vbox)
    self.vbox.Fit(self)

  def draw(self):
    print(self.comparisons)
    self.ax1.clear()
    self.ax1.imshow(self.images[self.comparisons[0][0]])
    self.ax1.axis('off')
    self.ax2.clear()
    self.ax2.imshow(self.images[self.comparisons[0][1]])
    self.ax2.axis('off')

  def on_left(self, event):
    self.rewards[self.comparisons[0][0]] += 1
    self.rewards[self.comparisons[0][1]] -= 1
    self.next_or_done()

  def on_right(self, event):
    self.rewards[self.comparisons[0][0]] -= 1
    self.rewards[self.comparisons[0][1]] += 1
    self.next_or_done()

  def on_equal(self, event):
    self.next_or_done()

  def next_or_done(self):
    self.comparisons.pop(0)
    if not self.comparisons:
      self.Destroy()
      wx.GetApp().ExitMainLoop()
    else:
      self.draw()
      self.canvas.draw()
      self.canvas.Refresh()

def gather_human_feedback(images):
  app = wx.App()
  app.frame = mainFrame(images)
  app.frame.Show()
  app.MainLoop()
  return app.frame.rewards

def main():
  # path to image data
  dataPath = '/Users/dwright/dev/gym-zooniverse/gym_zooniverse/envs/assets/'
  file = os.path.join(dataPath+'a01/a01-000u/a01-000u-00-01.png')
  image = get_training_example(file, plot=False)
  model = build_model(summary=True)
  npop=10
  alpha=0.001
  sigma=0.1
  niters = 100
  print('With npop=%d there will be %d comparisons (%dC2)'%(npop, n_choose_k(npop, 2), npop))
  # get the weights from this network (it is currently only randomly initialised)
  w = model.get_weights()
  for i in range(niters):
    images, w_tries = get_es_guesses(image, w, npop=npop, plot=False)
    rewards = gather_human_feedback(images)
    R = []
    for i in range(len(w_tries)):
      R.append(rewards[i])
    R = np.asarray(R)
    print(R)
    A = (R - np.mean(R)) / np.std(R)
    print(A)
    for i in range(len(w_tries)):
      for j in range(len(w_tries[i])):
        w[j] += alpha/(npop*sigma) * w_tries[i][j]*A[i]

if __name__ == '__main__':
  main()
