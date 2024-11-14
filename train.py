import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import torch
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

#from google.colab.patches import cv2_imshow

from model import Network
from configs import ModelConfigs

# Download and extract dataset
def download_and_unzip(url, extract_to='Datasets', chunk_size=1024*1024):
  http_response = urlopen(url)

  data = b''
  iterations = http_response.length // chunk_size + 1
  for _ in tqdm(range(iterations)):
    data += http_response.read(chunk_size)

  zipfile = ZipFile(BytesIO(data))
  zipfile.extractall(path=extract_to)

dataset_path = os.path.join('Datasets', 'IAM_Words')
if not os.path.exists(dataset_path):
  download_and_unzip('https://git.io/J0fjL', extract_to='Datasets')

  file = tarfile.open(os.path.join(dataset_path, "words.tgz"))
  file.extractall(os.path.join(dataset_path, "words"))

'''
Data preprocessing

This piece of code performs data preprocessing by parsing a `words.txt` file and populating three variables: `dataset`, `vocab`, and `max_len`. The dataset is a list that contains lists. Each inner list comprises a file path and its label. The vocab is a set of unique characters present in the labels. The `max_len` is the maximum length of the labels.

For each line in the file, the code executes the following tasks:

1. Skips the line if it starts with #;
2. Skips the line if the second element after splitting the line by space is "err";
3. Extracts the first three and eight characters of the filename and label, respectively;
4. Joins the dataset_path with the extracted folder names and filenames to form the file path;
5. Skips the line if the file path does not exist;
6. Otherwise, it adds the file path and `label` to the dataset list. Additionally, it updates the vocab set with the characters present in the label and updates the `max_len` variable to hold the maximum value of the current `max_len` and the length of the label;
After pre-processing the dataset, the code saves the vocabulary and the maximum text length to configs using the `ModelConfigs` class.
'''

dataset, vocab, max_len = [], set(), 0

# Preprocess the dataset by the specific IAM_Words dataset file structure
words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
  if line.startswith("#"):
    continue

  line_split = line.split(" ")
  if line_split[1] == "err":
    continue

  folder1 = line_split[0][:3]
  folder2 = "-".join(line_split[0].split("-")[:2])
  file_name = line_split[0] + ".png"
  label = line_split[-1].rstrip('\n')

  rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
  if not os.path.exists(rel_path):
    print(f"File not found: {rel_path}")
    continue

  dataset.append([rel_path, label])
  vocab.update(list(label))
  max_len = max(max_len, len(label))

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
  dataset=dataset,
  skip_validation=True,
  batch_size=configs.batch_size,
  data_preprocessors=[ImageReader(CVImage)],
  transformers=[
    #ImageShowCV2(), # uncomment to show images during training
    ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
    LabelIndexer(configs.vocab),
    LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
  use_cache=True,
)

#for _ in data_provider:
#  pass

# Split the dataset into training and validation sets
train_dataProvider, test_dataProvider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
    ]

network = Network(len(configs.vocab), activation="leaky_relu", dropout=0.3)
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# uncomment to print network summary, torchsummaryX package is required
summary(network, torch.zeros((1, configs.height, configs.width, 3)))