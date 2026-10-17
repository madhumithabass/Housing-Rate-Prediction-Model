# MASTER PROMPT: BUILD A PRODUCTION-READY BOSTON HOUSING PRICE PREDICTION WEB APP

## ROLE

Act as a senior machine learning engineer, Python backend engineer, full-stack developer, UI/UX designer, application security engineer, and DevOps engineer.

Build a complete, production-oriented housing price prediction web application using Flask and my existing trained artificial neural network regression model:

ANN_regression.h5

The project must be implemented directly in the current workspace.

Do not merely provide instructions, mockups, pseudocode, or a static frontend. Build the actual application, integrate the supplied model, implement all functional features, test the prediction workflow, and provide reproducible deployment instructions.

The application should be visually impressive enough for a professional portfolio, machine learning demonstration, or technical presentation.

Use sensible defaults and proceed autonomously unless a genuine blocker prevents implementation.

---

# 1. INSPECT THE EXISTING TRAINED MODEL FIRST

Before implementing prediction logic, inspect the supplied ANN_regression.h5 file and any existing training scripts, notebooks, datasets, preprocessing objects, or project documentation in the workspace.

The saved model configuration indicates a Sequential neural network with:

- Input shape: 13 features.
- Hidden Dense layer: 64 neurons, ReLU activation.
- Hidden Dense layer: 32 neurons, ReLU activation.
- Hidden Dense layer: 16 neurons, ReLU activation.
- Output layer: 1 neuron, linear activation.
- Training loss: mean squared error.
- Training metric: mean absolute error.

Verify these details directly from the model artifact rather than relying exclusively on this description.

The model appears consistent with a Boston Housing-style regression problem. Do not assume that the dataset, feature order, preprocessing, target units, or output scaling are confirmed solely by the architecture.

Investigate the following:

1. Determine whether the model can be loaded with the installed TensorFlow/Keras version.
2. Verify its input and output shapes.
3. Inspect the saved model configuration and layer structure.
4. Search for StandardScaler, MinMaxScaler, fitted preprocessing pipelines, or other transformations