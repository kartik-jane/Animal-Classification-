# Animal Classification Project

This project is an animal classification system built using deep learning. It uses a pre-trained model to classify images of various animals into their respective categories.

## Features

- Classify images of animals into 90 different categories
- Web-based interface using Streamlit
- Pre-trained Keras model for efficient inference
- Easy-to-use prediction script

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd animal-classification
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web App
Run the Streamlit web application:
```
streamlit run appp.py
```
This will start a local web server where you can upload images and get classification results.

### Command Line
Use the prediction script:
```
python src/predict.py --image_path path/to/your/image.jpg
```

### Training
To retrain the model:
```
python src/train.py
```

## Project Structure

- `animals/`: Dataset of animal images organized by category
- `model/`: Pre-trained model files
  - `animal_model.h5`: Keras model
  - `classes.json`: Class labels
- `src/`: Source code
  - `predict.py`: Prediction script
  - `train.py`: Training script
  - `utils.py`: Utility functions
- `outputs/`: Output directory for results
- `main.py`: Main entry point
- `requirements.txt`: Python dependencies
- `notes.txt`: Project notes

## Model

The model is a convolutional neural network trained on a dataset of animal images. It achieves high accuracy in classifying animals from the provided categories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.