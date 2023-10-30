from keras.models import load_model
import numpy as np

model_path = r"src/garbage/garbage_classification_model.h5"

model = load_model(model_path)

def classify_garbage(image):

  img = image.resize((224, 224))
  
  img = np.asarray(img)
  
  img = np.expand_dims(img, axis=0)
  
  predictions = model.predict(img)
  
  pred = np.argmax(predictions)

  confidence = predictions[0][pred]

  if confidence > 0.5:
    return "Not Garbage"
  else:
    return "Garbage"