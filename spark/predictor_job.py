import os
import pickle
from stream_predict_pandas import PandasPredictor  # New Pandas-only version

def main():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    predictor = PandasPredictor(model_dir)
    predictor.run()

if __name__ == "__main__":
    main()