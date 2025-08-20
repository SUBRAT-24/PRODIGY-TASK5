# Food Recognition & Calorie Estimator

A web application that recognizes food items from images and estimates their calorie content using deep learning.

## Features

- **Food Recognition**: Uses a pre-trained MobileNetV2 model to identify food items in images
- **Calorie Estimation**: Provides estimated calorie content based on recognized food items
- **Portion Size Adjustment**: Allows users to specify portion sizes for more accurate calorie estimates
- **Responsive Design**: Works on both desktop and mobile devices
- **Drag & Drop Interface**: Intuitive interface for uploading food images

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd food-calorie-estimator
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## How to Use

1. **Upload an Image**:
   - Click on the upload area or drag and drop a food image
   - Supported formats: JPG, JPEG, PNG

2. **Set Portion Size** (optional):
   - Adjust the portion size in grams (default is 100g)
   - The calorie estimate will be scaled based on the portion size

3. **View Results**:
   - The application will display the recognized food items
   - For each item, you'll see:
     - Food name
     - Estimated calories
     - Confidence level
     - Portion size

## Supported Food Items

The application currently recognizes the following food items:
- Apple
- Banana
- Hamburger
- Pizza
- Sandwich
- Salad
- French Fries
- Ice Cream
- Donut
- Hot Dog

## Limitations

- The accuracy depends on the quality of the input image
- The model may not recognize all food items or may misclassify similar-looking foods
- Calorie estimates are based on average values and may not be 100% accurate

## Future Improvements

- Add more food categories to the recognition model
- Implement user accounts to track food intake over time
- Add barcode scanning for packaged foods
- Include nutritional information beyond just calories
- Improve portion size estimation using reference objects

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses TensorFlow and Keras for deep learning
- MobileNetV2 model pre-trained on ImageNet
- Frontend built with Tailwind CSS
- Flask for the web server
