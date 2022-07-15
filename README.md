# Deep Learning

AI program code with Python. In this project, I developed a command line application to classify images.
I provide a Jupyter notebook with the code work.

Tools used in this project:

- PyTorch
- ArgParse
- Jason
- PIL
- NumPy
- Pandas
- matplotlib
- scikit-learn 


Command Line Application

Basic usage: python predict.py /path/to/image model

Options:
* Return top K most likely classes: python predict.py /path/to/image saved_model --top_k K
* Use a mapping categories names: python predict.py /path/to/image saved_model --category_names map.json

Example:
python predict.py ./test_images/orchid.jpg keras_model.h5 --top_k 3 --category_names label_map.json
