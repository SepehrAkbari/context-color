## Data

There were two datasets used in this project to train the models. The first dataset is intended to train a general model to correctly detect a vast variety of shapes, while the second dataset is used to train a more specific model, intended to detect faces and facial features.

### COCO Dataset

The COCO dataset (2017 subset) was used to train a general model, containing about 118,000 images, it is a suitable dataset for training a model to detect a wide variety of objects, spaces, and even to some extent people.

COCO offers a wide range of datasets at [COCO Dataset](https://cocodataset.org/). Their data is available at [COCO Dataset Image](http://images.cocodataset.org/). You can download any of the images, and use them for training. The dataset we used was the 2017 training subset, which you can download by running the following command:

```bash
cd data
mkdir -p coco/images
cd coco/images
curl -O http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
cd ../../../
```

To fetch the annotations, in JSON format, you can run the following command:

```bash
cd data
mkdir - p coco/annotations
cd coco/annotations
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ../../../
```

### CelebA Dataset

The large-scale CelebFaces attributes (CelebA) dataset with more than 200,000 celebrity images, each with 40 attribute annotations is widely used for face detection and recognition tasks. It is also a great option for training a model to detect faces and facial features.

This dataset is by Multimedia Laboratory, The Chinese University of Hong Kong. You can find the dataset at [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The dataset is available for download on Kaggle. You can either directly download it, [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), or use the Kaggle API to download it. To use the Kaggle API, you need to create an account on Kaggle and generate an API token, you have to then place the `kaggle.json` file in the `~/.kaggle/` directory. You can do this by running the following commands:

```bash
mkdir mkdir ~/.kaggle
mv <location>/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

After that, download the dataset by running the following command:

```bash
cd data
mkdir -p celeba
cd celeba
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip
cd ../../../
```