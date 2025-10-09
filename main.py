from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel 

# 1. Initialize the FastAPI app
app = FastAPI()

# --- Before you run this, you need to create your model.pkl file ---
# --- I see the code for it in the last cell of your notebook.    ---
# --- Just uncomment it and run that cell to create the file.     ---

# 2. Load your trained model
# Make sure the 'model.pkl' file is in the same directory
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please ensure the model file is in the correct directory.")
    model = None # Set model to None to avoid further errors

# 3. Create a Pydantic model for the input data
# This defines the structure and data types for your API's input
class CropFeatures(BaseModel):
    N: int
    P: int
    K: int
    TEMP: float
    PH: float
    RAINFALL: float

# 4. Create the decoding map (number -> crop name)
# This mapping comes directly from how the LabelEncoder worked in your notebook.
# NOTE: You will need to generate and save your encoder object to create this mapping.
# I have added a cell to your notebook to do this. For now, I've created a sample map.
CROP_NAMES_MAP = {
    0: 'Adzuki Beans',
    1: 'Arecanut',
    2: 'Arhar/Tur',
    3: 'Bajra',
    4: 'Bajra ',
    5: 'Barley',
    6: 'Black gram',
    7: 'Brinjal',
    8: 'Bullock Heart',
    9: 'Cabbage',
    10: 'Cashewnuts',
    11: 'Castor seed',
    12: 'Cauliflower',
    13: 'Chickpea',
    14: 'Coconut',
    15: 'Coconut ',
    16: 'Coffee',
    17: 'Coriander',
    18: 'Cotton',
    19: 'Cotton(lint)',
    20: 'Dry chillies',
    21: 'Dry ginger',
    22: 'Garlic',
    23: 'Ginger',
    24: 'Gram',
    25: 'Ground Nut',
    26: 'Groundnut',
    27: 'Groundnut ',
    28: 'Horse-gram',
    29: 'Jowar',
    30: 'Jowar ',
    31: 'Jute',
    32: 'Jute & mesta',
    33: 'Kidney Beans',
    34: 'Korra',
    35: 'Lemon',
    36: 'Lentil',
    37: 'Linseed',
    38: 'Linseed ',
    39: 'Maize',
    40: 'Mango',
    41: 'Mango ',
    42: 'Masoor',
    43: 'Moong(Green Gram)',
    44: 'Moth Beans',
    45: 'Mung Bean',
    46: 'Niger seed',
    47: 'Oilseeds total',
    48: 'Onion',
    49: 'Other Cereals & Millets',
    50: 'Other Kharif pulses',
    51: 'Other Rabi pulses',
    52: 'Paddy',
    53: 'Peas',
    54: 'Pigeon Peas',
    55: 'Potato',
    56: 'Ragi',
    57: 'Ragi ',
    58: 'Rapeseed &Mustard',
    59: 'Rice',
    60: 'Rubber',
    61: 'Safflower',
    62: 'Sannhamp',
    63: 'Sesamum',
    64: 'Small millets',
    65: 'Soyabean',
    66: 'Sugarcane',
    67: 'Sunflower',
    68: 'Sweet potato',
    69: 'Tapioca',
    70: 'Tea',
    71: 'Tobacco',
    72: 'Turmeric',
    73: 'Urad',
    74: 'Varagu',
    75: 'Wheat',
    76: 'apple',
    77: 'banana',
    78: 'black gram',
    79: 'chickpea',
    80: 'coffee',
    81: 'cotton',
    82: 'grapes',
    83: 'ground nut',
    84: 'jute',
    85: 'kidney beans',
    86: 'lentil',
    87: 'maize',
    88: 'mango',
    89: 'millet',
    90: 'moth beans',
    91: 'mung bean',
    92: 'muskmelon',
    93: 'orange',
    94: 'paddy',
    95: 'papaya',
    96: 'peas',
    97: 'pigeon peas',
    98: 'pomegranate',
    99: 'pulses',
    100: 'rice',
    101: 'rubber',
    102: 'sugarcane',
    103: 'tea',
    104: 'tobacco',
    105: 'watermelon',
    106: 'wheat'
}
    # ... and so on for all 109 of your crops.
    # You will need to generate the full map.

# 5. Define the prediction endpoint
@app.post("/predict")
def predict_crop(features: CropFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check the server logs."}
        
    # Convert the input data into a numpy array in the correct order
    # Your model was trained on ['N', 'P', 'K', 'PH', 'RAINFALL', 'TEMP']
    # The order MUST be the same.
    input_data = np.array([[
        features.N,
        features.P,
        features.K,
        features.PH,
        features.RAINFALL,
        features.TEMP
    ]])

    # Get the numeric prediction from the model
    numeric_prediction = model.predict(input_data)[0]

    # Convert the number to the actual crop name using our map
    crop_name = CROP_NAMES_MAP.get(int(numeric_prediction), "Unknown Crop")

    # Return the result
    return {"predicted_crop": crop_name}