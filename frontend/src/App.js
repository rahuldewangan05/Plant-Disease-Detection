import React, { useState } from "react";
import axios from "axios";

const PlantSelection = ({ onSelect }) => {
  const plants = ['Apple', 'Potato', 'Tomato', 'Corn'];
  
  return (
    <div className="bg-white p-6 rounded-lg shadow-md w-96 text-center">
      <h2 className="text-xl font-semibold mb-4">Select a Plant</h2>
      <div className="grid grid-cols-2 gap-3">
        {plants.map((p) => (
          <button
            key={p}
            className="bg-blue-500 text-white px-4 py-3 rounded-lg hover:bg-blue-600 transition duration-200"
            onClick={() => onSelect(p.toLowerCase())}
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
};

const ImageUpload = ({ plant, onUpload }) => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!image) {
      setError("Please upload an image.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", image);
    formData.append("plant", plant);

    try {
      console.log("Sending request to server...");
      const response = await axios.post(
        "http://127.0.0.1:5000/predict", 
        formData, 
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 60000 // 60 seconds timeout - models can take time to load
        }
      );
      
      console.log("Server response:", response.data);
      
      if (response.data.error) {
        setError(response.data.error);
      } else {
        onUpload(response.data);
      }
    } catch (error) {
      console.error("Error predicting:", error);
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Error data:", error.response.data);
        console.error("Error status:", error.response.status);
        setError(error.response.data?.error || `Server error: ${error.response.status}`);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("No response received:", error.request);
        setError("No response from server. Please check if the backend is running.");
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error("Error message:", error.message);
        setError(`Error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg shadow-md w-96 text-center">
      <h2 className="text-xl font-semibold mb-4">
        Upload an Image for {plant.charAt(0).toUpperCase() + plant.slice(1)}
      </h2>
      
      <div className="mb-4 border-2 border-dashed border-gray-300 p-4 rounded-lg">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageChange} 
          className="hidden" 
          id="image-upload" 
        />
        <label 
          htmlFor="image-upload" 
          className="cursor-pointer text-blue-500 hover:text-blue-700"
        >
          {imagePreview ? "Change Image" : "Select Image"}
        </label>
        
        {imagePreview && (
          <div className="mt-4">
            <img 
              src={imagePreview} 
              alt="Preview" 
              className="max-h-48 max-w-full mx-auto rounded-lg" 
            />
          </div>
        )}
      </div>
      
      {error && <p className="text-red-500 mb-4">{error}</p>}
      
      <button 
        type="submit" 
        className="bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 transition duration-200 w-full disabled:opacity-50"
        disabled={loading || !image}
      >
        {loading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Predicting...
          </span>
        ) : (
          "Upload & Predict"
        )}
      </button>
    </form>
  );
};

const PredictionResults = ({ results, onReset }) => {
  const { plant, predictions, available_classes } = results;
  
// Check if we have any successful predictions
const hasValidPredictions = Object.values(predictions).some(
  p => typeof p === 'object' && p.disease && p.disease !== 'Unknown'
);

if (!hasValidPredictions) {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md w-96 text-center">
      <h2 className="text-xl font-semibold mb-4">No Valid Predictions</h2>
      <p className="text-gray-700 mb-4">
        We couldn't generate valid predictions for this image. This could be due to:
      </p>
      <ul className="list-disc text-left pl-8 mb-4">
        <li>Image quality issues</li>
        <li>The disease may not be recognized by our system</li>
        <li>Technical problems with the prediction models</li>
      </ul>
      <button 
        className="bg-blue-500 text-white px-4 py-3 rounded-lg hover:bg-blue-600 transition duration-200 w-full"
        onClick={() => onReset(true)}
      >
        Try Another Plant/Image
      </button>
    </div>
  );
}

  return (
    <div className="bg-white p-6 rounded-lg shadow-md w-96 text-center">
      <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
      <p className="text-gray-700 mb-4">
        Plant: <span className="font-semibold">{plant.charAt(0).toUpperCase() + plant.slice(1)}</span>
      </p>
      
      <div className="space-y-4">
        {Object.entries(predictions).map(([model, result]) => (
          <div key={model} className="bg-gray-50 p-3 rounded-lg">
            <h3 className="text-lg font-medium text-gray-800">{model}</h3>
            {typeof result === 'object' ? (
              <div>
                <p className="text-gray-700">
                  Disease: <span className="font-semibold">{result.disease}</span>
                </p>
                <p className="text-gray-700">
                  Confidence: <span className="font-semibold">{result.confidence}</span>
                </p>
              </div>
            ) : (
              <p className="text-gray-700">{result}</p>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-6 grid grid-cols-2 gap-3">
        <button 
          className="bg-blue-500 text-white px-4 py-3 rounded-lg hover:bg-blue-600 transition duration-200"
          onClick={onReset}
        >
          Try Another Image
        </button>
        <button 
          className="bg-gray-500 text-white px-4 py-3 rounded-lg hover:bg-gray-600 transition duration-200"
          onClick={() => onReset(true)}
        >
          Change Plant
        </button>
      </div>
    </div>
  );
};

const App = () => {
  const [plant, setPlant] = useState("");
  const [results, setResults] = useState(null);

  const handleReset = (changePlant = false) => {
    setResults(null);
    if (changePlant) {
      setPlant("");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Plant Disease Detection</h1>
      
      {!plant ? (
        <PlantSelection onSelect={setPlant} />
      ) : !results ? (
        <ImageUpload plant={plant} onUpload={setResults} />
      ) : (
        <PredictionResults results={results} onReset={handleReset} />
      )}
    </div>
  );
};

export default App;