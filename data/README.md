```bash
mkdir mkdir ~/.kaggle
mv <location>/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

```bash
cd data
mkdir -p celeba
cd celeba
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip
```