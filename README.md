# lldt
Experiments for: *Streaming Decision Trees for Lifelong Learning*.

### Code
- HT/HT-AE, IRF/IRF-AE, BAG, RSP: *learners/ht.py*, *learners/irf.py*

- Evaluation: *benchmark/lldt_runner.py*

### Packages
```
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

### Extractors
Can be downloaded from: https://drive.google.com/drive/folders/1bNv8s3QXrZWMy1PTs_4B2TGTeM_HzI21?usp=sharing

### Run

If you use TensorBoard: 

```
tensorboard --logdir runs/lldt
```

Then:
```
python main.py
```

### Results

- CSV files can be found in *results*.

- TensorBoard: http://localhost:6006/

### Citation
```
@InProceedings{Korycki:2021sdl,
  author="Korycki, Lukasz
  and Krawczyk, Bartosz",
  title="Streaming Decision Trees for Lifelong Learning",
  booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
  year="2021",
  pages="502--518",
  isbn="978-3-030-86486-6"
}
```
